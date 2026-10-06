/* AEGIS · live 3D spacecraft — procedural, zero external assets.
   Every visual is driven by the telemetry stream: wheel spin, array
   sun-tracking, battery SOC, heater glow, dish→ground-station downlink,
   orbit position and the day/night terminator. Faults light the physical
   subsystem that produced them; channel z-scores drive per-part heat. */
(function () {
  "use strict";
  const C = window.CRAFT = { booted: false, ok: true, peek: false };
  let THREE, renderer, scene, camera, sunL, fillL, hemi, ambL;
  let earthMesh, earthGill, station, stationGlow, orbitLine, stars;
  let orbitGrp, craftTilt, busG, battFill, battShell, heaterGlow;
  let wheelSpokes = [], wheelsGrp, arrayPanes = [], radiator, payAp, payGlow;
  let obcLeds = [], dish, dishAim, beamMesh, beamMat, pulseRings = [];
  let sensorLeds = [], thrusterPlumes = [], thrusterVis = false;
  let EARTH_POS = null;
  const parts = {};                 // key -> {obj,obj2,mats:[],chans:[]}
  let live = null, visible = false, autoOrbit = true, drag = null, focusPart = null;
  const cam = { yaw: .7, pitch: .32, dist: 11.5, tYaw: .7, tPitch: .32, tDist: 11.5,
                cx: 0, cy: 0, cz: 0, shake: 0 };
  let twinkle = 0, lastT = 0, lastEvId = null, hudTick = 0;
  const EARTH_R = 11, ORBIT_R = 17.4;
  const EARTH = { x: -30, y: -26, z: -8 };
  const SUN_DIR = { x: .84, y: .28, z: .12 };

  const CH2PART = {
    battery_voltage: "battery", battery_current: "battery", soc: "battery", battery_temp: "battery",
    solar_current: "arrays", heater_current: "heater", panel_temp: "radiator", radiator_temp: "radiator",
    wheel_speed: "wheels", wheel_current: "wheels", wheel_temp: "wheels", attitude_err: "wheels",
    signal_strength: "comm", payload_power: "payload", payload_temp: "payload", cpu_load: "obc",
  };
  const FAULT2PART = {
    heater_stall: "heater", cell_sag: "battery", thermal_runaway: "battery",
    wheel_friction: "wheels", payload_overload: "payload", sensor_drift: "sensors",
    sensor_stuck: "sensors", sensor_bias: "sensors", link_loss: "comm",
  };
  const UNIT = { soc: "%", battery_voltage: " V", battery_current: " A", battery_temp: " °C",
    solar_current: " A", heater_current: " A", panel_temp: " °C", radiator_temp: " °C",
    wheel_speed: " rpm", wheel_current: " A", wheel_temp: " °C", attitude_err: "°",
    signal_strength: " dBm", payload_power: " W", payload_temp: " °C", cpu_load: "%" };
  const PART_CH = {
    battery: ["soc", "battery_temp", "battery_voltage"],
    wheels: ["wheel_speed", "wheel_temp", "attitude_err"],
    arrays: ["solar_current"], heater: ["heater_current"],
    radiator: ["panel_temp", "radiator_temp"],
    payload: ["payload_power", "payload_temp"], comm: ["signal_strength"],
    obc: ["cpu_load"],
  };
  const PART_LABEL = { battery: "BATTERY / PDU", wheels: "REACTION WHEELS", arrays: "SOLAR ARRAYS",
    heater: "BATTERY HEATER", radiator: "RADIATOR", payload: "PAYLOAD", comm: "X-BAND COMM", obc: "OBC" };
  C.CH2PART = CH2PART; C.PART_LABEL = PART_LABEL; C.UNIT = UNIT; C.PART_CH = PART_CH; C.FAULT2PART = FAULT2PART;
  C.fmt = (ch, v) => v == null ? "—" : (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2)) + (UNIT[ch] || "");

  const clamp = (x, a, b) => x < a ? a : x > b ? b : x;
  const smooth = (a, b, x) => { x = clamp((x - a) / (b - a), 0, 1); return x * x * (3 - 2 * x); };
  const mat = (c, o) => new THREE.MeshStandardMaterial(Object.assign({ color: c, metalness: .3, roughness: .55 }, o || {}));
  const edge = (geo, c, op) => new THREE.LineSegments(new THREE.EdgesGeometry(geo),
    new THREE.LineBasicMaterial({ color: c || 0x7fd7ff, transparent: true, opacity: op == null ? .45 : op }));
  const A = {};
  const aSet = (k, v) => { if (A[k] == null) A[k] = { cur: v, t: v }; else A[k].t = v; };
  const aStep = (k, dt, rate) => { const a = A[k]; if (!a) return 0;
    a.cur += (a.t - a.cur) * (1 - Math.exp(-dt * (rate || 6))); return a.cur; };

  /* ---------------- procedural earth texture ---------------- */
  function earthTexture() {
    try {
    const cv = document.createElement("canvas"); cv.width = 1024; cv.height = 512;
    const g = cv.getContext("2d"); if (!g || !g.createLinearGradient || !g.beginPath) return null;
    const grd = g.createLinearGradient(0, 0, 0, 512);
    grd.addColorStop(0, "#0e3a5c"); grd.addColorStop(.5, "#155078"); grd.addColorStop(1, "#0e3a5c");
    g.fillStyle = grd; g.fillRect(0, 0, 1024, 512);
    let s = 7; const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
    g.fillStyle = "#2f6b4f";
    for (let i = 0; i < 26; i++) {          // continent blobs
      const cx = rnd() * 1024, cy = 90 + rnd() * 330, r = 26 + rnd() * 88;
      for (let b = 0; b < 22; b++) {
        const a = rnd() * 6.283, d = rnd() * r, rr = r * (.24 + rnd() * .42);
        g.beginPath(); g.ellipse((cx + Math.cos(a) * d) % 1024, cy + Math.sin(a) * d * .62, rr, rr * (.5 + rnd() * .5), a, 0, 6.283);
        g.fill();
      }
    }
    g.fillStyle = "rgba(228,240,248,.85)";  // polar caps
    g.fillRect(0, 0, 1024, 34); g.fillRect(0, 478, 1024, 34);
    g.fillStyle = "rgba(255,255,255,.05)";  // cloud bands
    for (let i = 0; i < 60; i++) { const x = rnd() * 1024, y = 60 + rnd() * 392;
      g.beginPath(); g.ellipse(x, y, 26 + rnd() * 90, 4 + rnd() * 9, rnd() * .6, 0, 6.283); g.fill(); }
    const t = new THREE.CanvasTexture(cv); return t;
    } catch (e) { return null; }
  }

  /* ---------------- build ---------------- */
  function build() {
    THREE = window.THREE;
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x070b12);
    camera = new THREE.PerspectiveCamera(42, 16 / 9, .1, 800);

    sunL = new THREE.DirectionalLight(0xfff2d0, 2.5);
    sunL.position.set(SUN_DIR.x * 90, SUN_DIR.y * 90, SUN_DIR.z * 90); scene.add(sunL);
    fillL = new THREE.DirectionalLight(0x3f6f9e, .85);
    fillL.position.set(-40, 18, -30); scene.add(fillL);
    hemi = new THREE.HemisphereLight(0x8fb6d9, 0x10151c, .55); scene.add(hemi);
    ambL = new THREE.AmbientLight(0x41586f, .8); scene.add(ambL);

    const ep = new THREE.Vector3(EARTH.x, EARTH.y, EARTH.z); EARTH_POS = ep;
    const eg = new THREE.SphereGeometry(EARTH_R, 48, 32);
    const tex = earthTexture();
    earthMesh = new THREE.Mesh(eg, new THREE.MeshStandardMaterial(
      tex ? { map: tex, roughness: .9, metalness: 0 } : { color: 0x1d4f74, roughness: .9 }));
    earthMesh.position.copy(ep); earthMesh.rotation.y = 1.9; scene.add(earthMesh);
    earthGill = new THREE.LineSegments(new THREE.WireframeGeometry(new THREE.SphereGeometry(EARTH_R + .07, 24, 16)),
      new THREE.LineBasicMaterial({ color: 0x59d3e8, transparent: true, opacity: .07 }));
    earthGill.position.copy(ep); scene.add(earthGill);
    const atmo = new THREE.Mesh(new THREE.SphereGeometry(EARTH_R + 1.15, 32, 24),
      new THREE.MeshBasicMaterial({ color: 0x6ec6ff, transparent: true, opacity: .06, side: THREE.BackSide }));
    atmo.position.copy(ep); scene.add(atmo);

    /* ground station (beam target) on earth's lit limb */
    const stLocal = new THREE.Vector3(EARTH_R * .62, EARTH_R * .52, EARTH_R * .62).normalize().multiplyScalar(EARTH_R + .1);
    station = new THREE.Vector3(EARTH.x, EARTH.y, EARTH.z).add(stLocal);
    stationGlow = new THREE.Mesh(new THREE.SphereGeometry(.22, 12, 8),
      new THREE.MeshBasicMaterial({ color: 0x66e0ff, transparent: true, opacity: .9 }));
    stationGlow.position.copy(station); scene.add(stationGlow);

    /* orbit path ellipse (tilted circle around earth) */
    orbitGrp = new THREE.Group(); orbitGrp.position.copy(new THREE.Vector3(EARTH.x, EARTH.y, EARTH.z));
    orbitGrp.rotation.x = .42; orbitGrp.rotation.z = .2; scene.add(orbitGrp);
    const pts = [];
    for (let i = 0; i <= 128; i++) { const a = i / 128 * Math.PI * 2;
      pts.push(new THREE.Vector3(Math.cos(a) * ORBIT_R, 0, Math.sin(a) * ORBIT_R)); }
    orbitLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({ color: 0x59d3e8, transparent: true, opacity: .18 }));
    orbitGrp.add(orbitLine);

    const sg = new THREE.BufferGeometry(), P = new Float32Array(1400 * 3), S = new Float32Array(1400);
    for (let i = 0; i < 1400; i++) {
      const v = new THREE.Vector3().randomDirection().multiplyScalar(260 + Math.random() * 160);
      P.set([v.x, v.y, v.z], i * 3); S[i] = .4 + Math.random() * 1.4;
    }
    sg.setAttribute("position", new THREE.BufferAttribute(P, 3));
    stars = new THREE.Points(sg, new THREE.PointsMaterial({ color: 0xd4dfeb, size: .9, transparent: true, opacity: .85 }));
    scene.add(stars);

    craftTilt = new THREE.Group(); orbitGrp.add(craftTilt);
    craftTilt.position.set(ORBIT_R, 0, 0);   // relative to orbit group centre

    /* ------- bus: open frame so internals are visible ------- */
    busG = new THREE.Group(); craftTilt.add(busG);
    const gold = mat(0xd9b13b, { metalness: .65, roughness: .32 });
    const plate = new THREE.BoxGeometry(2.6, .16, 2.0);
    const deckLo = new THREE.Mesh(plate, gold); deckLo.position.y = -.95;
    const deckHi = new THREE.Mesh(plate, gold); deckHi.position.y = .95;
    busG.add(deckLo, deckHi, edge(plate, 0x8fe0f2, .55), edge(plate, 0x8fe0f2, .55));
    deckHi.position.copy(new THREE.Vector3(0, .95, 0)); deckLo.position.copy(new THREE.Vector3(0, -.95, 0));
    [[-1.15, -0.9], [1.15, -0.9], [-1.15, .9], [1.15, .9]].forEach(c => {
      const post = new THREE.Mesh(new THREE.BoxGeometry(.14, 1.9, .14), gold);
      post.position.set(c[0], 0, c[1]); busG.add(post); });

    /* battery — transparent shell + SOC fill, standing on lower deck */
    const batG = new THREE.Group(); batG.position.set(-.55, -.55, 0); busG.add(batG);
    battShell = new THREE.Mesh(new THREE.BoxGeometry(1.3, .82, 1.4),
      new THREE.MeshStandardMaterial({ color: 0xbfd0e0, transparent: true, opacity: .22, roughness: .15, metalness: .1 }));
    const fgeo = new THREE.BoxGeometry(1.14, 1, 1.26); fgeo.translate ? fgeo.translate(0, .5, 0) : 0;
    battFill = new THREE.Mesh(fgeo, mat(0x2f9e44, { emissive: 0x14531f, emissiveIntensity: .5 }));
    battFill.position.y = -.41; battFill.scale.y = .7;
    battShell.add(edge(battShell.geometry, 0x9fb8cc, .5));
    batG.add(battShell, battFill);
    for (let i = 1; i < 4; i++) {           // tick marks on shell
      const tk = new THREE.Mesh(new THREE.BoxGeometry(1.32, .015, .1), mat(0x9fb4c8, { transparent: true, opacity: .6 }));
      tk.position.set(0, -.41 + i * .205, .68); batG.add(tk);
    }

    /* heater loop wrapping the battery */
    const hg = new THREE.TorusGeometry(.86, .055, 10, 44);
    heaterGlow = new THREE.Mesh(hg, new THREE.MeshStandardMaterial({ color: 0x6b4a26, emissive: 0xff6a00, emissiveIntensity: 0, roughness: .4 }));
    heaterGlow.rotation.x = Math.PI / 2; heaterGlow.position.copy(batG.position); heaterGlow.position.y += .28;
    busG.add(heaterGlow);

    /* reaction wheels — exposed on top deck, spoked */
    const wheelMats = [];
    wheelsGrp = new THREE.Group(); wheelsGrp.position.set(.62, .48, 0); busG.add(wheelsGrp);
    [[-.66, 0, "x"], [0, 0, "y"], [.66, 0, "z"]].forEach((w, i) => {
      const stand = new THREE.Mesh(new THREE.BoxGeometry(.14, .34, .14), mat(0x77838f));
      stand.position.set(w[0], -.16, w[1]); wheelsGrp.add(stand);
      const gg = new THREE.CylinderGeometry(.3, .3, .34, 24);
      const gm = new THREE.Mesh(gg, mat(0x9aa8b8, { metalness: .6, roughness: .3, emissive: 0x000000 }));
      if (w[2] === "x") gm.rotation.z = Math.PI / 2;
      gm.position.set(w[0], .12, w[1]);
      const hub = new THREE.Mesh(new THREE.CylinderGeometry(.07, .07, .38, 10), mat(0xd64545, { emissive: 0x611, emissiveIntensity: .4 }));
      wheelMats.push(gm.material, hub.material);
      hub.rotation.copy(gm.rotation); gm.add(hub);
      const spokes = new THREE.Group(); spokes.rotation.copy(gm.rotation);   // rotates
      for (let k = 0; k < 3; k++) {
        const sp = new THREE.Mesh(new THREE.BoxGeometry(.62, .028, .05), mat(0xe8eef4, { metalness: .4 }));
        sp.rotation.y = k * Math.PI / 3; spokes.add(sp); wheelMats.push(sp.material);
      }
      spokes.position.copy(gm.position);
      wheelsGrp.add(gm, spokes); wheelSpokes.push(spokes);
    });

    /* solar wings — deploy from yokes, single-axis sun tracking */
    const mkWing = sgn => {
      const yoke = new THREE.Group(); yoke.position.set(sgn * 1.34, 0, 0); busG.add(yoke);
      const boom = new THREE.Mesh(new THREE.CylinderGeometry(.05, .05, 1.0, 8), mat(0x8090a0, { metalness: .6 }));
      boom.rotation.z = Math.PI / 2; boom.position.x = sgn * .5; yoke.add(boom);
      const tiltn = new THREE.Group(); tiltn.position.x = sgn * 1.0; yoke.add(tiltn);
      for (let p = 0; p < 3; p++) {
        const pg = new THREE.BoxGeometry(1.62, .05, 1.12);
        const pm = new THREE.MeshStandardMaterial({ color: 0x123a63, metalness: .2, roughness: .14,
          emissive: 0x2f8fd6, emissiveIntensity: .12 });
        const pane = new THREE.Mesh(pg, pm);
        pane.position.set(sgn * (p * 1.66), 0, 0);
        tiltn.add(pane, edge(pg, 0x6fd9ee, .5));
        arrayPanes.push(pm);
      }
      const back = new THREE.Mesh(new THREE.BoxGeometry(4.7, .02, 1.08), mat(0x233043, { metalness: .1 }));
      back.position.set(sgn * 1.64, -.045, 0); tiltn.add(back);
      return { yoke, tilt: tiltn, sgn };
    };
    const wingL = mkWing(-1), wingR = mkWing(1);

    /* radiator — hinged fin panel, back-top */
    radiator = new THREE.Group(); busG.add(radiator);
    const fin = new THREE.BoxGeometry(1.9, .06, 1.05);
    const finM = new THREE.MeshStandardMaterial({ color: 0xe6ecf3, metalness: .15, roughness: .55, emissive: 0x21435e, emissiveIntensity: 0 });
    const finMesh = new THREE.Mesh(fin, finM);
    for (let i = 0; i < 6; i++) { const rib = new THREE.Mesh(new THREE.BoxGeometry(1.86, .1, .04), mat(0xc6d2de));
      rib.position.set(0, .05, -.42 + i * .17); finMesh.add(rib); }
    finMesh.position.set(0, 1.36, -.86); finMesh.rotation.x = -.32;
    radiator.add(finMesh); radiator.userData.mat = finM;

    /* payload — telescope under the bus, aperture to nadir */
    const payG = new THREE.Group(); payG.position.set(-.1, -1.5, 0); busG.add(payG);
    const pgeo = new THREE.CylinderGeometry(.42, .42, .92, 22);
    payG.add(new THREE.Mesh(pgeo, mat(0x9a7a22, { metalness: .55, roughness: .35 })), edge(pgeo, 0xffd166, .6));
    payAp = new THREE.Mesh(new THREE.CylinderGeometry(.3, .3, .1, 22),
      new THREE.MeshStandardMaterial({ color: 0x0b0f14, emissive: 0x2fd06a, emissiveIntensity: 0 }));
    payAp.position.y = -.5; payG.add(payAp);
    payGlow = new THREE.Mesh(new THREE.ConeGeometry(.6, 2.2, 20, 1, true),
      new THREE.MeshBasicMaterial({ color: 0x59ffa0, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false }));
    payGlow.position.y = -1.7; payG.add(payGlow);

    /* OBC cards with LED rows */
    const obcG = new THREE.Group(); obcG.position.set(1.05, -.35, 0); busG.add(obcG);
    obcG.add(new THREE.Mesh(new THREE.BoxGeometry(.12, 1.15, 1.42), mat(0x1f3d31, { roughness: .8 })));
    for (let r = 0; r < 2; r++) for (let i = 0; i < 6; i++) {
      const led = new THREE.Mesh(new THREE.BoxGeometry(.05, .07, .09),
        new THREE.MeshStandardMaterial({ color: 0x2c4c40, emissive: 0x64ffa8, emissiveIntensity: .08 }));
      led.position.set(.09, .42 - i * .17, r ? .3 + i * .01 : -.42 + i * .03);
      obcG.add(led); obcLeds.push(led);
    }

    /* comm dish on gimbal + downlink beam and pulse rings */
    dish = new THREE.Group(); busG.add(dish);
    dish.position.set(.3, 1.35, .55);
    const mast = new THREE.Mesh(new THREE.CylinderGeometry(.05, .05, .5, 8), mat(0x8090a0));
    mast.position.y = -.25; dish.add(mast);
    dishAim = new THREE.Group(); dish.add(dishAim);
    const dg = new THREE.CylinderGeometry(.52, .09, .3, 22, 1, true);
    const dishM = new THREE.MeshStandardMaterial({ color: 0xeef2f6, metalness: .25, roughness: .45, side: THREE.DoubleSide });
    const dishMesh = new THREE.Mesh(dg, dishM); dishMesh.rotation.x = Math.PI / 2;
    const feed = new THREE.Mesh(new THREE.CylinderGeometry(.018, .018, .6, 6), mat(0x9aa6b4));
    feed.rotation.x = Math.PI / 2; feed.position.z = .26;
    dishAim.add(dishMesh, feed, new THREE.Mesh(new THREE.SphereGeometry(.05, 8, 6), mat(0xffcf6a, { emissive: 0x7a4c00, emissiveIntensity: .8 })));
    parts.dishMat = dishM;

    beamMat = new THREE.MeshBasicMaterial({ color: 0x66e0ff, transparent: true, opacity: .1, side: THREE.DoubleSide, depthWrite: false });
    const bg = new THREE.ConeGeometry(1.15, 16, 26, 1, true);
    if (bg.translate) bg.translate(0, -8, 0);          // apex at origin, opens along -Y
    beamMesh = new THREE.Mesh(bg, beamMat); scene.add(beamMesh);
    for (let i = 0; i < 3; i++) {
      const r = new THREE.Mesh(new THREE.TorusGeometry(.5, .035, 6, 22),
        new THREE.MeshBasicMaterial({ color: 0x9ef0ff, transparent: true, opacity: .0 }));
      scene.add(r); pulseRings.push(r);
    }

    /* sensor pods + RCS thrusters */
    for (let i = 0; i < 3; i++) {
      const led = new THREE.Mesh(new THREE.SphereGeometry(.07, 10, 8),
        new THREE.MeshStandardMaterial({ color: 0xc8d2dc, emissive: 0xffb020, emissiveIntensity: 0 }));
      led.position.set(.7 - i * .7, -.8, .95); busG.add(led); sensorLeds.push(led);
    }
    [[1, -1.05, .8], [-1, -1.05, .8], [1, -1.05, -.8], [-1, -1.05, -.8]].forEach(p => {
      const nz = new THREE.Mesh(new THREE.CylinderGeometry(.09, .14, .2, 10), mat(0x6a7683));
      nz.position.set(p[0], p[1], p[2]); busG.add(nz);
      const fl = new THREE.Mesh(new THREE.ConeGeometry(.11, .5, 10),
        new THREE.MeshBasicMaterial({ color: 0x9fdcff, transparent: true, opacity: 0, blending: THREE.AdditiveBlending || THREE.NormalBlending, depthWrite: false }));
      fl.position.set(p[0], p[1] - .34, p[2]); busG.add(fl); thrusterPlumes.push(fl);
    });

    parts.bus = { obj: busG, mats: [gold], chans: [] };
    parts.battery = { obj: batG, mats: [battFill.material], chans: PART_CH.battery };
    parts.wheels = { obj: wheelsGrp, mats: wheelMats, chans: PART_CH.wheels };
    parts.heater = { obj: heaterGlow, mats: [heaterGlow.material], chans: PART_CH.heater };
    parts.arrays = { obj: wingL.yoke, obj2: wingR.yoke, mats: arrayPanes, chans: PART_CH.arrays };
    parts.radiator = { obj: finMesh, mats: [finM], chans: PART_CH.radiator };
    parts.payload = { obj: payG, mats: [payAp.material], chans: PART_CH.payload };
    parts.obc = { obj: obcG, mats: obcLeds.map(l => l.material), chans: PART_CH.obc };
    parts.comm = { obj: dish, mats: [dishM], chans: PART_CH.comm };
    parts.sensors = { obj: busG, mats: sensorLeds.map(l => l.material), chans: [] };
    C._wings = [wingL, wingR];

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    if ("outputEncoding" in THREE && THREE.sRGBEncoding !== undefined) renderer.outputEncoding = THREE.sRGBEncoding;
  }

  /* ---------------- input ---------------- */
  function bindInput(canvas) {
    canvas.addEventListener("pointerdown", e => {
      drag = { x: e.clientX, y: e.clientY, yaw: cam.tYaw, pitch: cam.tPitch };
      try { canvas.setPointerCapture(e.pointerId); } catch (err) {}
      autoOrbit = false;
    });
    canvas.addEventListener("pointermove", e => {
      if (!drag) return;
      cam.tYaw = drag.yaw - (e.clientX - drag.x) * .0055;
      cam.tPitch = clamp(drag.pitch + (e.clientY - drag.y) * .0045, -1.15, 1.15);
    });
    window.addEventListener("pointerup", () => drag = null);
    canvas.addEventListener("wheel", e => {
      e.preventDefault();
      cam.tDist = clamp(cam.tDist * (1 + Math.sign(e.deltaY) * .11), 3.5, 40);
    }, { passive: false });
    const rc = new THREE.Raycaster(), m = new THREE.Vector2();
    let downAt = null;
    canvas.addEventListener("pointerdown", e => downAt = [e.clientX, e.clientY]);
    canvas.addEventListener("click", e => {
      if (downAt && Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5) return; // was a drag
      const r = canvas.getBoundingClientRect();
      m.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      rc.setFromCamera(m, camera);
      const hits = rc.intersectObject(craftTilt, true);
      let found = null;
      if (hits.length) {
        let o = hits[0].object;
        while (o && !found) { for (const k in parts) if (parts[k].obj === o || parts[k].obj2 === o) { found = k; break; } o = o.parent; }
        if (!found) found = nearestPart(hits[0].point);
      }
      focusPart = found; C.onFocus && C.onFocus(found);
      if (found) framePart(found, true);
    });
  }
  function nearestPart(p) {
    let best = null, bd = 1e9; const t = new THREE.Vector3();
    for (const k in parts) { const q = parts[k].obj; if (!q || !q.getWorldPosition) continue;
      const d = q.getWorldPosition(t).distanceTo(p);
      if (d < bd) { bd = d; best = k; } }
    return bd < 2.2 ? best : null;
  }
  function framePart(k, fromClick) {
    if (k && (!parts[k] || !parts[k].obj || !parts[k].obj.getWorldPosition)) return;
    cam.tDist = fromClick ? 6.4 : 7.6;
    if (k && parts[k].obj.getWorldPosition) {   // swing to the part's outward face
      const q = parts[k].obj.getWorldPosition(new THREE.Vector3());
      const c = craftTilt.getWorldPosition(new THREE.Vector3());
      const nx = q.x - c.x, ny = q.y - c.y, nz = q.z - c.z;
      cam.tYaw = Math.atan2(nz, nx);
      cam.tPitch = clamp(Math.atan2(ny, Math.hypot(nx, nz)) + .3, -.2, 1.0);
    }
  }

  /* ---------------- data -> targets ---------------- */
  function setTargets() {
    if (!live || live.warming) return;
    const v = live.values || {}, z = live.z || [], chans = live.channels || [];
    const Z = c => { const i = chans.indexOf(c); return i < 0 ? 0 : Math.abs(z[i] || 0); };
    const num = (c, d) => { const x = v[c]; return x == null || !isFinite(x) ? d : +x; };
    aSet("phase", (live.phase != null ? live.phase : 0) * Math.PI * 2);
    aSet("soc", clamp((num("soc", 70) - 15) / 83, 0, 1));
    aSet("batT", clamp(num("battery_temp", 10) / 30, 0, 1));
    aSet("solar", clamp(num("solar_current", 0) / 40, 0, 1));
    aSet("heat", clamp(num("heater_current", 0) / 2.4, 0, 1));
    aSet("wheelrpm", clamp(Math.abs(num("wheel_speed", 0)) / 60, 0, 1.4));
    aSet("att", clamp(Z("attitude_err") / 4, 0, 1.5));
    aSet("sig", clamp((num("signal_strength", 40) - 20) / 25, 0, 1));
    aSet("pay", clamp(num("payload_power", 0) / 45, 0, 1));
    aSet("cpu", clamp(num("cpu_load", 0) / 100, 0, 1));
    aSet("radT", clamp((num("radiator_temp", -5) + 20) / 60, 0, 1));
    for (const k in parts) {
      const p = parts[k]; if (!p.chans) continue;
      let h = 0; p.chans.forEach(c => h = Math.max(h, Z(c)));
      aSet("heat_" + k, clamp((h - 1.5) / 3, 0, 1));
    }
    const last = live.last || {};
    const hot = {};
    (last.parts || []).forEach(c => { const k = CH2PART[c]; if (k) hot[k] = last.origin === "DATA" ? .4 : .95; });
    (live.active || []).forEach(f => { const k = FAULT2PART[f]; if (k) hot[k] = 1; });
    for (const k in parts) aSet("fault_" + k, hot[k] || 0);
    aSet("dump", (last.actions || []).includes("momentum_dump_plan") && last.live ? 1 : 0);
    if (last.id && last.id !== lastEvId) { lastEvId = last.id; cam.shake = last.live ? .5 : 0; }
    if (last.live) {
      const hottest = Object.keys(hot).sort((a, b) => hot[b] - hot[a])[0];
      if (hottest && focusPart !== hottest) { focusPart = hottest; C.onFocus && C.onFocus(hottest); framePart(hottest); }
    }
  }

  /* ---------------- frame ---------------- */
  const tmp1 = () => new THREE.Vector3();
  function frame(now) {
    requestAnimationFrame(frame);
    if (!C.ok || !C.booted) return;
    const dt = clamp((now - (lastT || now)) / 1000, .001, .05); lastT = now;
    if (!visible && !C.peek) return;
    twinkle += dt;

    const ph = aStep("phase", dt, 7);
    craftTilt.position.set(Math.cos(ph) * ORBIT_R, 0, Math.sin(ph) * ORBIT_R);
    const cw = tmp1(); if (craftTilt.getWorldPosition) craftTilt.getWorldPosition(cw);
    if (typeof cw.x !== "number") { cw.x = Math.cos(ph) * ORBIT_R; cw.y = 0; cw.z = Math.sin(ph) * ORBIT_R; }
    const ex = EARTH.x - cw.x, ey = EARTH.y - cw.y, ez = EARTH.z - cw.z;   // craft → earth
    const SL = Math.hypot(SUN_DIR.x, SUN_DIR.y, SUN_DIR.z);
    const EL = Math.hypot(ex, ey, ez) || 1;
    const align = (SUN_DIR.x * ex + SUN_DIR.y * ey + SUN_DIR.z * ez) / (SL * EL);
    const ecl = smooth(-.16, .02, align);               // 1 when earth blocks the sun
    sunL.intensity = 2.5 * (1 - .92 * ecl);
    ambL.intensity = .8 - .28 * ecl; hemi.intensity = .55 - .3 * ecl;
    earthMesh.rotation.y += dt * .006;

    const att = aStep("att", dt, 4), wf = aStep("fault_wheels", dt, 5);
    const rpm = aStep("wheelrpm", dt, 3);
    wheelSpokes.forEach((sp, i) => { sp.rotation.y += dt * (2 + 30 * rpm + wf * 26) * (i === 1 ? -1 : 1); });
    craftTilt.rotation.z = Math.sin(twinkle * 7.1) * .013 * (att + .18 + wf);
    craftTilt.rotation.x = Math.sin(twinkle * 5.3) * .016 * (att + .18 + wf * 2);
    craftTilt.rotation.y = Math.sin(twinkle * .35) * .06 + (att > .4 ? Math.sin(twinkle * 3) * .05 : 0);

    const soc = aStep("soc", dt, 2.2), bt = aStep("batT", dt, 3);
    battFill.scale.y = Math.max(.05, .82 * soc);
    const badHue = clamp(.35 - .35 * bt - .3 * aStep("fault_battery", dt, 4), 0, .35);
    battFill.material.color.setHSL(badHue, .62, .42);
    battFill.material.emissive.setHSL(badHue, .7, .12);
    heaterGlow.material.emissiveIntensity = aStep("heat", dt, 4) * (.5 + 2.2 * ecl)
      + aStep("fault_heater", dt, 5) * 1.7 * (.5 + .5 * Math.sin(twinkle * 7));

    // sun-tracking arrays: wings tilt through the orbit so panels keep the sun
    const s = aStep("solar", dt, 3);
    C._wings.forEach(w => {
      const trk = Math.sin(ph + w.sgn * .35) * .85 * (1 - ecl);   // stow flat in eclipse
      w.tilt.rotation.x += (trk - w.tilt.rotation.x) * (1 - Math.exp(-dt * 2.5));
    });
    arrayPanes.forEach(m => { m.emissiveIntensity = (ecl > .5 ? .02 : .06) + s * 1.05 * (1 - ecl); });

    radiator.userData.mat.emissiveIntensity = aStep("radT", dt, 3) * .85;
    payAp.material.emissiveIntensity = aStep("pay", dt, 4) * 1.7 * (ecl > .4 ? 1 : .55);
    payGlow.material.opacity = aStep("pay", dt, 4) * .10 * (1 - ecl * .5);
    const cpu = aStep("cpu", dt, 5);
    obcLeds.forEach((b, i) => { b.material.emissiveIntensity = (cpu * 12 > i * 1.6 + Math.sin(twinkle * 6 + i)) ? 1.5 : .08; });

    // dish tracks the ground station; beam cone + travelling pulse rings
    if (dishAim && station) {
      const dwp = tmp1(); if (dish.getWorldPosition) dish.getWorldPosition(dwp);
      if (dishAim.lookAt) dishAim.lookAt(station);
      const sig = aStep("sig", dt, 3) * (1 - clamp(aStep("fault_comm", dt, 5), 0, 1));
      beamMat.opacity = .03 + sig * .22;
      if (beamMesh.position.set) beamMesh.position.set(dwp.x, dwp.y, dwp.z);
      if (beamMesh.lookAt) { beamMesh.lookAt(station); beamMesh.rotateX(Math.PI / 2); }
      const L = Math.hypot(station.x - dwp.x, station.y - dwp.y, station.z - dwp.z) || 16;
      if (beamMesh.scale.set) beamMesh.scale.set(.8 + sig * .5, L / 16, .8 + sig * .5);
      pulseRings.forEach((r, i) => {
        const t = ((twinkle * (.5 + sig * 1.4) + i / 3) % 1);
        if (r.position.set) r.position.set(dwp.x + (station.x - dwp.x) * t, dwp.y + (station.y - dwp.y) * t, dwp.z + (station.z - dwp.z) * t);
        if (r.scale.set) r.scale.set(.6 + t * 2.6, .6 + t * 2.6, .6 + t * 2.6);
        if (r.material) r.material.opacity = sig * .5 * Math.sin(Math.PI * t);
        if (r.lookAt) r.lookAt(EARTH_POS);
      });
    }
    stationGlow.material.opacity = .5 + .45 * Math.sin(twinkle * 2.2);
    thrusterVis = aStep("dump", dt, 8) > .02;
    thrusterPlumes.forEach(f => { f.material.opacity = thrusterVis ? .35 + .45 * Math.abs(Math.sin(twinkle * 26)) : 0; });

    // per-part fault heat (explicit so wheel mats stay reachable)
    const pulse = .55 + .45 * Math.sin(twinkle * 7);
    for (const k in parts) {
      const p = parts[k]; if (!p.mats || !p.mats.length) continue;
      const lvl = Math.max(aStep("fault_" + k, dt, 5) * pulse, aStep("heat_" + k, dt, 5) * .5);
      p.mats.forEach(m => m && m.emissive && m.emissive.setRGB(lvl * .95, lvl * .12, lvl * .05));
    }
    wheelSpokes.forEach((sp, i) => sp.children.forEach(c => c.material && c.material.emissive &&
      c.material.emissive.setRGB(aStep("fault_wheels", dt, 5) * pulse * .9, 0, 0)));

    // data pathologies → sensor pods blink amber in sync with the stream
    const dp = (live && live.dp) || [];
    sensorLeds.forEach((led, i) => led.material.emissiveIntensity =
      dp.length ? ((twinkle * 4 + i) % 2 < 1 ? 1.4 : .1) : .0);

    // camera: eased orbit + target fly-to + event shake
    if (autoOrbit && !drag) cam.tYaw += dt * .1;
    const f = 1 - Math.exp(-dt * 5.5);
    cam.yaw += (cam.tYaw - cam.yaw) * f; cam.pitch += (cam.tPitch - cam.pitch) * f; cam.dist += (cam.tDist - cam.dist) * f;
    let tx, ty, tz;
    const fp = focusPart && parts[focusPart] && parts[focusPart].obj;
    const anchor = (fp || craftTilt);
    if (anchor.getWorldPosition) { const q = anchor.getWorldPosition(tmp1()); tx = q.x; ty = q.y; tz = q.z; }
    if (typeof tx !== "number") { tx = 0; ty = 0; tz = 0; }
    const kf = 1 - Math.exp(-dt * 6);
    cam.cx += (tx - cam.cx) * kf; cam.cy += (ty - cam.cy) * kf; cam.cz += (tz - cam.cz) * kf;
    if (cam.shake > .001) {
      cam.shake *= Math.exp(-dt * 3);
      cam.cx += Math.sin(twinkle * 61) * cam.shake * .08;
      cam.cy += Math.cos(twinkle * 53) * cam.shake * .08;
    }
    camera.position.set(cam.cx + cam.dist * Math.cos(cam.pitch) * Math.cos(cam.yaw),
      cam.cy + cam.dist * Math.sin(cam.pitch), cam.cz + cam.dist * Math.cos(cam.pitch) * Math.sin(cam.yaw));
    if (camera.lookAt) camera.lookAt(cam.cx, cam.cy, cam.cz);
    renderer.render(scene, camera);
    C._hud && C._hud.update((hudTick++ % 3) === 0);
  }

  /* ---------------- HUD: svg overlay with leader lines ---------------- */
  C.attachHud = function (svgEl, cardEl) {
    if (!svgEl) return;
    const NS = "http://www.w3.org/2000/svg";
    const mkG = k => {
      const g = document.createElementNS ? document.createElementNS(NS, "g") : { style: {}, setAttribute() {}, appendChild() {}, children: [] };
      g.setAttribute("class", "clabel");
      const line = document.createElementNS ? document.createElementNS(NS, "line") : { setAttribute() {} };
      const dot = document.createElementNS ? document.createElementNS(NS, "circle") : { setAttribute() {} };
      dot.setAttribute("r", "3");
      const box = document.createElementNS ? document.createElementNS(NS, "g") : { style: {}, setAttribute() {}, appendChild() {} };
      const bg = document.createElementNS ? document.createElementNS(NS, "rect") : { setAttribute() {} };
      bg.setAttribute("rx", "5"); bg.setAttribute("class", "lb-box");
      const t1 = document.createElementNS ? document.createElementNS(NS, "text") : { textContent: "", setAttribute() {} };
      t1.setAttribute("class", "lk");
      const t2 = document.createElementNS ? document.createElementNS(NS, "text") : { textContent: "", setAttribute() {} };
      t2.setAttribute("class", "lv");
      box.appendChild(bg); box.appendChild(t1); box.appendChild(t2);
      g.appendChild(line); g.appendChild(dot); g.appendChild(box);
      svgEl.appendChild(g);
      g.style.cursor = "pointer";
      g.addEventListener && g.addEventListener("click", () => C.camNudge(focusPart === k ? "reset" : k));
      return { g, line, dot, box, bg, t1, t2, k };
    };
    const LAYOUT = { battery: [-150, 96], wheels: [120, -110], arrays: [-190, -80], comm: [150, 60],
                    heater: [-170, -30], payload: [140, 130] };
    const labels = Object.keys(LAYOUT).map(mkG);
    C._hud = {
      update(txt) {
        if (!camera || !renderer || !live || live.warming) { labels.forEach(l => l.g.setAttribute && l.g.setAttribute("display", "none")); return; }
        const r = renderer.domElement.getBoundingClientRect(), W = r.width || 100, H = r.height || 100;
        labels.forEach((l, idx) => {
          const p = parts[l.k]; if (!p || !p.obj || !p.obj.getWorldPosition) return;
          const w = p.obj.getWorldPosition(new THREE.Vector3()).project(camera);
          if (w.z > 1 || !isFinite(w.x)) { l.g.setAttribute("display", "none"); return; }
          const ax = (w.x + 1) / 2 * W, ay = (1 - w.y) / 2 * H;
          const side = LAYOUT[l.k];
          const bx = clamp(ax + side[0], 8, W - 200), by = clamp(ay + side[1], 52, H - 64);
          l.g.setAttribute("display", "");
          l.line.setAttribute("x1", ax.toFixed(1)); l.line.setAttribute("y1", ay.toFixed(1));
          l.line.setAttribute("x2", (bx + (side[0] < 0 ? 186 : 10)).toFixed(1)); l.line.setAttribute("y2", (by + 20).toFixed(1));
          l.dot.setAttribute("cx", ax.toFixed(1)); l.dot.setAttribute("cy", ay.toFixed(1));
          l.box.setAttribute("transform", `translate(${bx.toFixed(1)},${by.toFixed(1)})`);
          l.bg.setAttribute("width", "186"); l.bg.setAttribute("height", "44");
          l.t1.setAttribute("x", "10"); l.t1.setAttribute("y", "16");
          l.t2.setAttribute("x", "10"); l.t2.setAttribute("y", "34");
          const fl = (live.active || []).some(f => FAULT2PART[f] === l.k);
          l.g.setAttribute("class", "clabel" + (fl ? " hot" : (focusPart === l.k ? " sel" : "")));
          if (txt) {
            const chs = p.chans || [];
            l.t1.textContent = PART_LABEL[l.k] || l.k;
            l.t2.textContent = chs.map(c => {
              const i = (live.channels || []).indexOf(c); const zz = i >= 0 ? (live.z[i] || 0) : 0;
              return `${c.split("_").slice(0, 2).join(" ")} ${C.fmt(c, live.values ? live.values[c] : null)}  z${zz >= 0 ? "+" : ""}${zz.toFixed(1)}`;
            }).join("   ");
          }
        });
      },
    };
    if (cardEl) C.onFocus = k => {
      if (!k) { cardEl.innerHTML = '<span class="note">click a subsystem — or a channel row on the Board — to focus it · drag orbits · wheel zooms</span>'; return; }
      const p = parts[k] || {};
      cardEl.innerHTML = `<b>${PART_LABEL[k] || k}</b><br>` + (p.chans || []).map(c => {
        const i = ((live && live.channels) || []).indexOf(c);
        const val = live && live.values ? live.values[c] : null;
        const z = i >= 0 && live ? live.z[i] : 0;
        const cls = Math.abs(z) >= 3 ? "hot" : Math.abs(z) >= 2 ? "warm" : "";
        return `<span class="mono">${c}</span> → <b>${C.fmt(c, val)}</b> <span class="cz ${cls}">z ${z}</span>`;
      }).join("<br>") + ((live && live.active || []).length ? `<br><span class="cz hot">faults: ${live.active.join(", ")}</span>` : "");
    };
  };

  /* ---------------- public ---------------- */
  C.boot = function (container) {
    if (C.booted) { C.resize(container); return C.ok; }
    if (!window.THREE) { C.ok = false; return false; }
    try { build(); } catch (e) { console.warn("3D unavailable:", e.message); C.ok = false; return false; }
    container.appendChild(renderer.domElement);
    renderer.domElement.style.display = "block";
    renderer.domElement.style.position = "absolute";
    renderer.domElement.style.inset = "0";
    bindInput(renderer.domElement);
    C.container = container; C.booted = true;
    C.resize(container);
    window.addEventListener("resize", () => C.resize(C.container));
    requestAnimationFrame(frame);
    return true;
  };
  C.resize = function (container) {
    container = container || C.container;
    if (!container || !C.ok || !C.booted) return;
    const w = container.clientWidth || window.innerWidth, h = container.clientHeight || window.innerHeight;
    renderer.setSize(w, h, false);
    renderer.domElement.style.width = w + "px"; renderer.domElement.style.height = h + "px";
    camera.aspect = w / h; camera.updateProjectionMatrix();
    C._hud && C._hud.update(true);
  };
  C.setLive = d => { const first = !live; live = d; setTargets();
    if (first && C.booted && craftTilt.getWorldPosition) { const q = craftTilt.getWorldPosition(new THREE.Vector3());
      cam.cx = q.x; cam.cy = q.y; cam.cz = q.z; } };
  C.setVisible = v => { visible = v; if (v) lastT = 0; };
  C.camNudge = kind => {
    if (kind === "reset") { focusPart = null; cam.tDist = 11.5; cam.tPitch = .32; cam.tYaw = cam.yaw + .6;
      C.onFocus && C.onFocus(null); }
    else if (parts[kind]) { focusPart = kind; framePart(kind, false); C.onFocus && C.onFocus(kind); }
    autoOrbit = false;
  };
  C.toggleAuto = () => { autoOrbit = !autoOrbit; return autoOrbit; };
  C.partsInfo = k => parts[k];
})();
