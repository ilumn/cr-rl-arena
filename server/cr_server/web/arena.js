// ===== arena board renderer =================================================
// Used by live human games and the replay viewer.
// Keep the drawing code in step with gym/viewer/index.html. Original vector drawing only.
//   board.draw({ t, ents:[{id,o,name,x,y,hp,f,hw,hh,a,hit,dx,dy,t0,t1}], events, deaths, hover, labels })
//   events: [[tick, seat, card, x, y]]   deaths: [{t,o,x,y,r,tower,king,hw,hh}]
//   f bits: 1 flying, 2 tower, 4 deploying.  dx,dy: movement to the next frame.  t0,t1: first seen, deploy done.
function createArena(cv) {
  const ctx = cv.getContext("2d");
  const PAD = 0.9;                      // decorated rim around the field, in tiles
  const TEAM = [{ main: "#5aa8ff", lite: "#a9d2ff", dark: "#1f4f8f" }, { main: "#ff6f91", lite: "#ffb3c4", dark: "#a02848" }];
  const OUT = "rgba(18,15,24,.92)", SKIN = "#f1c9a2", BONE = "#f1ecdc", WOOD = "#9a6a3c", STEEL = "#c3ccdb";
  // clothing / hide colour by what the unit is; first match wins
  const BODY = [
    [/skeleton|guard|bone/, "#d9d3c0"], [/bat/, "#7a5fb8"], [/goblin|spear/, "#8a5a34"], [/barbarian|berserker/, "#8a6642"],
    [/pekka/, "#3d4f80"], [/ice|snow|frost/, "#a8e0f5"], [/electro|zap|spark|tesla/, "#62c6f2"], [/elixir/, "#d070e0"],
    [/lava|fire|inferno|furnace|phoenix/, "#f08a2e"], [/golem|cannon|mortar|bomb|rock/, "#9a9cab"], [/giant/, "#c98a52"],
    [/dragon/, "#5fb862"], [/minion/, "#6a86e0"], [/witch|night|ghost|bandit/, "#8a5fc4"], [/wizard|magic/, "#d9603a"],
    [/hog|ram|rider|lumber|hunter/, "#8f5a34"], [/balloon|rocket/, "#e0564a"], [/archer|princess|musket|dart|firecracker/, "#e06fa6"],
    [/knight|prince|valkyrie|recruit|monk|champion|miner|royal/, "#8e9bb3"], [/heal|battle/, "#f2d55a"],
  ];
  // what to draw for it; first match wins
  const KIND = [
    [/tombstone/, { k: "tomb" }],
    [/cannon|mortar/, { k: "turret", top: "barrel" }], [/x_?bow/, { k: "turret", top: "bow" }],
    [/tesla/, { k: "turret", top: "orb", orb: "#8fe4ff" }], [/inferno_tower/, { k: "turret", top: "orb", orb: "#ff8a2e", tall: 1.5 }],
    [/bomb_tower/, { k: "turret", top: "orb", orb: "#2b2b33", tall: 1.5 }],
    [/goblin_hut|goblin_cage|goblin_drill/, { k: "hut", roof: "#5fae48" }], [/barbarian_hut/, { k: "hut", roof: "#d9a94a" }],
    [/furnace/, { k: "hut", roof: "#e0672a" }], [/collector|pump/, { k: "hut", roof: "#c75fd6" }], [/hut|cage|drill/, { k: "hut", roof: "#8d6b45" }],
    [/balloon|skeleton_barrel/, { k: "balloon" }], [/dragon|lava_hound|phoenix/, { k: "dragon" }], [/lava_pup|minion|bat/, { k: "imp" }],
    [/spirit/, { k: "spirit" }], [/hog|ram_rider|battle_ram/, { k: "mount", mount: "#b0703f", hat: "hair", hair: "#3a2a20" }],
    [/pekka/, { k: "human", bulk: 1.25, skin: "#4f66a3", hat: "horns", weapon: "sword", eye: "#ff7ad9" }],
    [/giant_skeleton/, { k: "human", bulk: 1.45, tall: 1.25, head: .9, skin: BONE, weapon: "none", eye: "#222" }],
    [/skeleton|guards|bone/, { k: "human", bulk: .8, skin: BONE, weapon: "sword", eye: "#222" }],
    [/goblin_giant/, { k: "human", bulk: 1.4, tall: 1.2, head: .9, skin: "#6cc552", ears: 1, weapon: "none" }],
    [/spear_goblin|dart_goblin/, { k: "human", bulk: .85, skin: "#6cc552", ears: 1, weapon: "spear" }],
    [/goblin/, { k: "human", bulk: .85, skin: "#6cc552", ears: 1, weapon: "sword" }],
    [/golem/, { k: "human", bulk: 1.5, tall: 1.2, head: .8, rock: 1, weapon: "none", eye: "#ffd24a" }],
    [/giant/, { k: "human", bulk: 1.45, tall: 1.2, head: .85, hat: "hair", hair: "#7a4a28", weapon: "none" }],
    [/wizard|witch|mother/, { k: "human", hat: "wizard", weapon: "staff" }],
    [/archer|princess|firecracker|hunter|magic/, { k: "human", hat: "hood", weapon: "bow" }],
    [/musketeer|sparky|zappies/, { k: "human", hat: "helmet", weapon: "gun" }],
    [/prince/, { k: "mount", mount: "#8d6b52", hat: "helmet" }],
    [/barbarian|berserker/, { k: "human", bulk: 1.1, hat: "hair", hair: "#e8bd55", weapon: "sword" }],
    [/knight|valkyrie|guard|recruit|champion|miner|bandit|ghost|monk|royal|lumber|executioner|bowler|ronin/, { k: "human", hat: "helmet", weapon: "sword" }],
  ];
  // spells get an area effect where they were cast: [radius in tiles, colour, seconds, style]
  const SPELL = {
    fireball: [2.5, "#ff9a3c", .8, "fire"], rocket: [2, "#ff7038", .9, "fire"], arrows: [4, "#e8e0c8", .9, "arrows"],
    zap: [2.5, "#8fdcff", .5, "bolt"], lightning: [3.5, "#fff3a0", .8, "bolt"], giant_snowball: [2.5, "#dff4ff", .7, "ice"],
    freeze: [3, "#9fe8ff", 4, "ice"], poison: [3.5, "#e08a2a", 8, "cloud"], rage: [3, "#c86bff", 6, "cloud"],
    heal: [3, "#ffe36b", 2, "cloud"], graveyard: [4, "#6fd6c0", 9, "cloud"], goblin_curse: [3, "#a6d85a", 6, "cloud"],
    void: [2.5, "#a070ff", 4, "cloud"], clone: [3, "#7fe9f0", .8, "cloud"], royal_delivery: [3, "#e8d08a", .8, "fire"],
    earthquake: [3.5, "#b08a5a", 3, "quake"], tornado: [5.5, "#cfcbe8", 1.1, "swirl"],
    the_log: [1.95, "#b07a45", 1.6, "roll", 10], log: [1.95, "#b07a45", 1.6, "roll", 10], barbarian_barrel: [1.3, "#b07a45", 1, "roll", 4.5],
  };
  const FLYERS = { dragon: 1, imp: 1, balloon: 1 };
  const specs = new Map();
  function spec(name, fly) {
    const key = name + (fly ? "^" : "");
    let v = specs.get(key);
    if (!v) {
      let cloth = null, kind = null;
      for (const [re, c] of BODY) if (re.test(name)) { cloth = c; break; }
      for (const [re, k] of KIND) if (re.test(name)) { kind = k; break; }
      if (!cloth) { let h = 0; for (const ch of name) h = (h * 31 + ch.charCodeAt(0)) % 360; cloth = `hsl(${h} 42% 56%)`; }
      if (!kind) kind = { k: "human", hat: "hair", hair: "#6b4a2e", weapon: "sword" };
      if (fly && !FLYERS[kind.k]) kind = { k: "imp" };
      v = Object.assign({ cloth }, kind);
      if (v.rock) v.skin = cloth;
      specs.set(key, v);
    }
    return v;
  }
  const abbr = (c) => { const w = String(c).split("_").filter(Boolean); return (w.length > 1 ? w[0][0] + w[1][0] : String(c).slice(0, 2)).toUpperCase(); };

  let meta = null, lanes = null, pads = [], tile = 20, pad = 12, W = 0, H = 0, dpr = 1, bg = null, shown = [];
  const X = (x) => pad + x * tile, Y = (y) => pad + (meta.h - y) * tile;
  const rr = (c, x, y, w, h, r) => { c.beginPath(); c.roundRect(x, y, w, h, Math.max(0, Math.min(r, w / 2, h / 2))); };
  const disc = (c, x, y, r) => { c.beginPath(); c.arc(x, y, r, 0, 7); };
  const rng = (seed) => { let s = (seed * 2654435761) >>> 0 || 1; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); };

  // towers: [{o, x, y, king, hw, hh}] as they stand at the start; lanes and paving follow them
  function configure(m, towers) {
    meta = m; pads = towers;
    const pr = [0, 1].map((o) => towers.filter((t) => t.o === o && !t.king));
    lanes = pr[0].length && pr[1].length
      ? { y0: pr[0].reduce((s, t) => s + t.y, 0) / pr[0].length, y1: pr[1].reduce((s, t) => s + t.y, 0) / pr[1].length } : null;
  }
  function resize(maxW, maxH) {
    tile = Math.max(8, Math.floor(Math.min(maxW / (meta.w + 2 * PAD), maxH / (meta.h + 2 * PAD))));
    pad = Math.round(PAD * tile);
    W = meta.w * tile + 2 * pad; H = meta.h * tile + 2 * pad; dpr = Math.min(2.5, devicePixelRatio || 1);
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); cv.style.width = W + "px"; cv.style.height = H + "px";
    paintStatic();
  }

  // ---------- the arena itself: painted once per size ----------
  function tree(c, x, y, r) {
    c.fillStyle = "rgba(0,0,0,.28)"; c.beginPath(); c.ellipse(x + r * .25, y + r * .45, r, r * .7, 0, 0, 7); c.fill();
    c.fillStyle = "#2f6e3a"; disc(c, x, y, r); c.fill(); c.strokeStyle = "#1c4526"; c.lineWidth = 1; c.stroke();
    c.fillStyle = "#3f8a48"; disc(c, x - r * .22, y - r * .25, r * .62); c.fill();
    c.fillStyle = "#5aa85a"; disc(c, x - r * .32, y - r * .36, r * .3); c.fill();
  }
  function paintStatic() {
    bg = document.createElement("canvas"); bg.width = cv.width; bg.height = cv.height;
    const c = bg.getContext("2d"); c.setTransform(dpr, 0, 0, dpr, 0, 0);
    const fw = meta.w * tile, fh = meta.h * tile, half = meta.river_half || 1, bh = meta.bridge_half || 1.5, rnd = rng(7);
    // woodland rim
    let g = c.createLinearGradient(0, 0, 0, H); g.addColorStop(0, "#27502f"); g.addColorStop(1, "#1f4329");
    c.fillStyle = g; c.fillRect(0, 0, W, H);
    for (let side = 0; side < 4; side++) {
      const len = side < 2 ? H : W, n = Math.round(len / (tile * 1.05));
      for (let i = 0; i <= n; i++) {
        const along = (i + (rnd() - .5) * .5) * (len / n), off = pad * (.34 + rnd() * .16), r = tile * (.3 + rnd() * .14);
        tree(c, side === 0 ? off : side === 1 ? W - off : along, side === 2 ? off : side === 3 ? H - off : along, r);
      }
    }
    // grass, two tones per tile, with tufts
    c.save(); c.beginPath(); c.rect(pad, pad, fw, fh); c.clip();
    for (let y = 0; y < meta.h; y++) for (let x = 0; x < meta.w; x++) {
      c.fillStyle = (x + y) & 1 ? "#69a84a" : "#74b553"; c.fillRect(pad + x * tile, pad + y * tile, tile, tile);
    }
    c.lineCap = "round"; c.lineWidth = Math.max(1, tile * .06);
    for (let i = 0; i < meta.w * meta.h * .45; i++) {
      const x = pad + rnd() * fw, y = pad + rnd() * fh, s = tile * (.1 + rnd() * .08);
      if (rnd() < .12) { c.fillStyle = rnd() < .5 ? "#fff6c9" : "#ffd9ec"; disc(c, x, y, Math.max(1, tile * .06)); c.fill(); continue; }
      c.strokeStyle = rnd() < .5 ? "rgba(40,90,30,.45)" : "rgba(170,225,120,.5)";
      c.beginPath(); c.moveTo(x - s, y); c.lineTo(x - s * .4, y - s * 1.3); c.moveTo(x, y); c.lineTo(x + s * .3, y - s * 1.6); c.moveTo(x + s, y); c.lineTo(x + s * 1.2, y - s * 1.1); c.stroke();
    }
    // dirt lanes: across each bridge, joined in front of each king tower
    if (lanes) {
      const xs = meta.bridges, x0 = Math.min(...xs), x1 = Math.max(...xs);
      const path = () => { c.beginPath();
        for (const bx of xs) { c.moveTo(X(bx), Y(lanes.y0)); c.lineTo(X(bx), Y(lanes.y1)); }
        c.moveTo(X(x0), Y(lanes.y0)); c.lineTo(X(x1), Y(lanes.y0)); c.moveTo(X(x0), Y(lanes.y1)); c.lineTo(X(x1), Y(lanes.y1)); };
      c.lineCap = "round"; c.lineJoin = "round";
      const lw = Math.min(2, bh * 1.4) * tile;
      c.strokeStyle = "rgba(40,70,25,.35)"; c.lineWidth = lw + tile * .5; path(); c.stroke();
      c.strokeStyle = "#a3854b"; c.lineWidth = lw + tile * .22; path(); c.stroke();
      c.strokeStyle = "#d6b876"; c.lineWidth = lw; path(); c.stroke();
      c.strokeStyle = "rgba(255,240,200,.16)"; c.lineWidth = lw * .5; path(); c.stroke();
      c.fillStyle = "rgba(120,92,48,.5)";               // pebbles
      for (let i = 0; i < 110; i++) {
        const v = rnd() < .62, bx = xs[Math.floor(rnd() * xs.length)], k = rnd(), j = (rnd() - .5) * lw * .8;
        const px = v ? X(bx) + j : X(x0 + (x1 - x0) * k), py = v ? Y(lanes.y0 + (lanes.y1 - lanes.y0) * k) : Y(rnd() < .5 ? lanes.y0 : lanes.y1) + j;
        disc(c, px, py, Math.max(.8, tile * (.03 + rnd() * .04))); c.fill();
      }
    }
    // paving under each tower
    for (const t of pads) {
      const hw = ((t.hw || 1.5) + .4) * tile, hh = ((t.hh || t.hw || 1.5) + .4) * tile, x = X(t.x), y = Y(t.y);
      c.fillStyle = "rgba(40,60,25,.3)"; rr(c, x - hw - 2, y - hh - 1, hw * 2 + 4, hh * 2 + 5, tile * .3); c.fill();
      c.fillStyle = "#b7b0a0"; rr(c, x - hw, y - hh, hw * 2, hh * 2, tile * .25); c.fill();
      c.save(); c.clip(); c.strokeStyle = "rgba(70,64,52,.4)"; c.lineWidth = 1; c.beginPath();
      const st = tile * .8;
      for (let py = y - hh, row = 0; py < y + hh; py += st, row++) { c.moveTo(x - hw, Math.round(py) + .5); c.lineTo(x + hw, Math.round(py) + .5);
        for (let px = x - hw + (row & 1 ? st / 2 : 0); px < x + hw; px += st) { c.moveTo(Math.round(px) + .5, py); c.lineTo(Math.round(px) + .5, py + st); } }
      c.stroke(); c.restore();
      c.strokeStyle = "#8a8474"; c.lineWidth = 1.5; rr(c, x - hw, y - hh, hw * 2, hh * 2, tile * .25); c.stroke();
    }
    // river with sand banks and a few rocks
    const ry = Y(meta.river_y + half), rh = half * 2 * tile, bank = Math.max(2, tile * .2);
    c.fillStyle = "#4a8436"; c.fillRect(pad, ry - bank, fw, rh + bank * 2);
    c.fillStyle = "#dcc98e"; c.fillRect(pad, ry - bank * .55, fw, rh + bank * 1.1);
    g = c.createLinearGradient(0, ry, 0, ry + rh); g.addColorStop(0, "#2f86d6"); g.addColorStop(.35, "#4aa5ec"); g.addColorStop(.7, "#59b2f2"); g.addColorStop(1, "#3a8fdc");
    c.fillStyle = g; c.fillRect(pad, ry, fw, rh);
    c.fillStyle = "rgba(8,40,100,.3)"; c.fillRect(pad, ry, fw, Math.max(1.5, tile * .14));
    c.fillStyle = "rgba(255,255,255,.42)"; c.fillRect(pad, ry + rh - Math.max(1.5, tile * .07), fw, Math.max(1.5, tile * .07));
    for (let i = 0; i < 7; i++) {
      const wx = rnd() * meta.w; if (meta.bridges.some((bx) => Math.abs(wx - bx) < bh + .6)) continue;
      const px = X(wx), py = ry + (rnd() < .5 ? rh * (.1 + rnd() * .12) : rh * (.8 + rnd() * .12)), s = tile * (.16 + rnd() * .14);
      c.fillStyle = "rgba(255,255,255,.35)"; c.beginPath(); c.ellipse(px, py + s * .5, s * 1.5, s * .55, 0, 0, 7); c.fill();
      c.fillStyle = "#8c93a0"; c.beginPath(); c.ellipse(px, py, s * 1.2, s * .8, 0, 0, 7); c.fill(); c.strokeStyle = "#565c68"; c.lineWidth = 1; c.stroke();
      c.fillStyle = "#b4bac6"; c.beginPath(); c.ellipse(px - s * .25, py - s * .22, s * .55, s * .3, 0, 0, 7); c.fill();
    }
    // plank bridges on stone footings
    for (const bx of meta.bridges) {
      const x = X(bx - bh), w = bh * 2 * tile, over = tile * .42, y = ry - over, h = rh + over * 2, rail = Math.max(2, tile * .22);
      c.fillStyle = "rgba(0,20,60,.3)"; c.fillRect(x + tile * .16, ry, w, rh);
      for (const fy of [y - tile * .1, y + h - tile * .26]) { c.fillStyle = "#9c978a"; rr(c, x - tile * .14, fy, w + tile * .28, tile * .36, 3); c.fill(); c.strokeStyle = "#625e54"; c.lineWidth = 1; c.stroke(); }
      c.fillStyle = "#b98249"; rr(c, x, y, w, h, 3); c.fill();
      const step = Math.max(4, tile * .4);
      for (let py = y, i = 0; py < y + h; py += step, i++) {
        if (i & 1) { c.fillStyle = "rgba(255,225,170,.14)"; c.fillRect(x + rail, py, w - rail * 2, Math.min(step, y + h - py)); }
        c.fillStyle = "rgba(80,45,15,.5)"; c.fillRect(x + rail, Math.round(py), w - rail * 2, 1);
      }
      for (const rx of [x, x + w - rail]) {
        c.fillStyle = "#74491f"; rr(c, rx, y, rail, h, 2); c.fill();
        c.fillStyle = "rgba(255,255,255,.18)"; c.fillRect(rx + 1, y + 1, Math.max(1, rail * .3), h - 2);
        for (const py of [y + 1, y + h / 2 - rail * .6, y + h - rail * 1.2 - 1]) { c.fillStyle = "#4f3013"; rr(c, rx - 1, py, rail + 2, rail * 1.2, 2); c.fill(); }
      }
    }
    // light from above, darker toward the corners
    g = c.createRadialGradient(pad + fw / 2, pad + fh / 2, fh * .25, pad + fw / 2, pad + fh / 2, fh * .72);
    g.addColorStop(0, "rgba(255,255,220,.05)"); g.addColorStop(1, "rgba(0,20,0,.24)");
    c.fillStyle = g; c.fillRect(pad, pad, fw, fh);
    c.restore();
    // low stone wall round the field
    const wt = Math.max(3, Math.round(tile * .2));
    c.fillStyle = "#8d897b"; c.beginPath(); c.rect(pad - wt, pad - wt, fw + wt * 2, fh + wt * 2); c.rect(pad, pad, fw, fh); c.fill("evenodd");
    c.strokeStyle = "rgba(40,36,28,.55)"; c.lineWidth = 1; c.beginPath();
    for (let x = 0; x <= meta.w; x++) { c.moveTo(X(x) + .5, pad - wt); c.lineTo(X(x) + .5, pad); c.moveTo(X(x) + .5, pad + fh); c.lineTo(X(x) + .5, pad + fh + wt); }
    for (let y = 0; y <= meta.h; y++) { c.moveTo(pad - wt, Y(y) + .5); c.lineTo(pad, Y(y) + .5); c.moveTo(pad + fw, Y(y) + .5); c.lineTo(pad + fw + wt, Y(y) + .5); }
    c.stroke();
    c.fillStyle = "rgba(255,255,255,.22)"; c.fillRect(pad - wt, pad - wt, fw + wt * 2, 1.5); c.fillRect(pad - wt, pad - wt, 1.5, fh + wt * 2);
    c.strokeStyle = "#3e3b33"; c.lineWidth = 1.5; c.strokeRect(pad - wt, pad - wt, fw + wt * 2, fh + wt * 2); c.strokeRect(pad, pad, fw, fh);
  }

  // ---------- pieces ----------
  function hpBar(x, y, w, h, hp, o) {
    ctx.fillStyle = "rgba(8,10,16,.9)"; rr(ctx, x - w / 2 - 1.5, y - 1.5, w + 3, h + 3, 3); ctx.fill();
    const fw = Math.max(1, w * Math.max(0, Math.min(1, hp)));
    ctx.fillStyle = TEAM[o].main; rr(ctx, x - w / 2, y, fw, h, 2); ctx.fill();
    ctx.fillStyle = "rgba(255,255,255,.34)"; ctx.fillRect(x - w / 2 + 1, y + .5, Math.max(0, fw - 2), Math.max(1, h * .3));
  }
  function crown(x, y, s) {
    ctx.beginPath();
    ctx.moveTo(x - s * .5, y + s * .3); ctx.lineTo(x - s * .5, y - s * .2); ctx.lineTo(x - s * .25, y + s * .04); ctx.lineTo(x, y - s * .36);
    ctx.lineTo(x + s * .25, y + s * .04); ctx.lineTo(x + s * .5, y - s * .2); ctx.lineTo(x + s * .5, y + s * .3); ctx.closePath();
    ctx.fillStyle = "#ffcf4a"; ctx.fill(); ctx.strokeStyle = "#7a4d00"; ctx.lineWidth = Math.max(1, s * .06); ctx.lineJoin = "round"; ctx.stroke();
    ctx.fillStyle = "#fff1b8"; ctx.fillRect(x - s * .42, y + s * .14, s * .84, s * .07);
  }
  const lw = (u) => Math.max(1, u * .085);

  // a standing figure, feet at the origin; u is roughly half its width
  function human(u, sp, tm, st) {
    const b = sp.bulk || 1, tall = sp.tall || 1, hs = (sp.head || 1) * .4 * u, back = st.back;
    const step = st.moving ? Math.sin(st.phase) : 0, bob = st.moving ? Math.abs(Math.cos(st.phase)) * u * .07 : 0;
    ctx.lineWidth = lw(u); ctx.strokeStyle = OUT; ctx.lineJoin = "round"; ctx.lineCap = "round";
    if (!st.legless) {
      ctx.fillStyle = sp.rock ? sp.cloth : "#3b3442";
      for (const d of [-1, 1]) { const up = Math.max(0, d * step) * u * .16; rr(ctx, d * .2 * b * u - .12 * u, -.36 * u - up, .24 * u, .36 * u, .08 * u); ctx.fill(); ctx.stroke(); }
    }
    const top = -bob - (.3 + .72 * tall) * u, hy = top - hs * .58, hand = b * .5 * u;
    const stick = (x1, y1, x2, y2, col, w) => { ctx.strokeStyle = OUT; ctx.lineWidth = w + lw(u) * 1.6; ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
      ctx.strokeStyle = col; ctx.lineWidth = w; ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); ctx.strokeStyle = OUT; ctx.lineWidth = lw(u); };
    const weapon = () => {
      const wy = -bob - .62 * u, w = sp.weapon;
      if (w === "sword") { stick(hand, wy, hand + .34 * u, wy - .95 * u, "#e9eef6", u * .13); stick(hand - .12 * u, wy - .12 * u, hand + .2 * u, wy - .02 * u, "#c9a13c", u * .1); }
      else if (w === "spear") { stick(hand, wy + .3 * u, hand + .2 * u, wy - 1.25 * u, WOOD, u * .09); ctx.fillStyle = "#e9eef6"; ctx.beginPath(); ctx.moveTo(hand + .2 * u - .13 * u, wy - 1.2 * u); ctx.lineTo(hand + .26 * u, wy - 1.62 * u); ctx.lineTo(hand + .36 * u, wy - 1.2 * u); ctx.closePath(); ctx.fill(); ctx.stroke(); }
      else if (w === "staff") { stick(hand, wy + .35 * u, hand + .12 * u, wy - 1.05 * u, WOOD, u * .1); ctx.fillStyle = sp.cloth; disc(ctx, hand + .13 * u, wy - 1.2 * u, u * .22); ctx.fill(); ctx.stroke(); ctx.fillStyle = "rgba(255,255,255,.6)"; disc(ctx, hand + .07 * u, wy - 1.27 * u, u * .08); ctx.fill(); }
      else if (w === "bow") { ctx.strokeStyle = OUT; ctx.lineWidth = u * .17; ctx.beginPath(); ctx.arc(hand - .05 * u, wy - .15 * u, u * .5, -1.15, 1.15); ctx.stroke(); ctx.strokeStyle = WOOD; ctx.lineWidth = u * .09; ctx.stroke();
        ctx.strokeStyle = "rgba(255,255,255,.75)"; ctx.lineWidth = Math.max(.75, u * .04); ctx.beginPath(); ctx.moveTo(hand - .05 * u + Math.cos(1.15) * u * .5, wy - .15 * u - Math.sin(1.15) * u * .5); ctx.lineTo(hand - .05 * u + Math.cos(1.15) * u * .5, wy - .15 * u + Math.sin(1.15) * u * .5); ctx.stroke(); ctx.strokeStyle = OUT; ctx.lineWidth = lw(u); }
      else if (w === "gun") { stick(hand - .15 * u, wy, hand + .7 * u, wy - .22 * u, "#4a4f5c", u * .15); }
    };
    if (back) weapon();
    // torso with a team sash
    ctx.fillStyle = sp.cloth; rr(ctx, -.45 * b * u, top, .9 * b * u, -bob - .28 * u - top, .24 * u); ctx.fill(); ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,.18)"; rr(ctx, -.34 * b * u, top + .08 * u, .3 * b * u, .2 * u * tall, .08 * u); ctx.fill();
    ctx.fillStyle = tm.main; rr(ctx, -.45 * b * u + lw(u) / 2, -bob - .52 * u, .9 * b * u - lw(u), .17 * u, .04 * u); ctx.fill();
    if (!back) weapon();
    // head
    if (sp.ears) { ctx.fillStyle = sp.skin; for (const d of [-1, 1]) { ctx.beginPath(); ctx.moveTo(d * hs * .7, hy - hs * .2); ctx.lineTo(d * hs * 1.75, hy - hs * .5); ctx.lineTo(d * hs * .8, hy + hs * .35); ctx.closePath(); ctx.fill(); ctx.stroke(); } }
    if (sp.hat === "horns") { ctx.fillStyle = "#dfe3ee"; for (const d of [-1, 1]) { ctx.beginPath(); ctx.moveTo(d * hs * .55, hy - hs * .6); ctx.lineTo(d * hs * 1.25, hy - hs * 1.6); ctx.lineTo(d * hs * .95, hy - hs * .2); ctx.closePath(); ctx.fill(); ctx.stroke(); } }
    ctx.fillStyle = sp.skin || SKIN; disc(ctx, 0, hy, hs); ctx.fill(); ctx.stroke();
    const cap = (col, from, to) => { ctx.fillStyle = col; ctx.beginPath(); ctx.arc(0, hy, hs, from, to); ctx.closePath(); ctx.fill(); ctx.stroke(); };
    if (sp.hat === "hair") cap(sp.hair, back ? 0 : Math.PI * 1.05, back ? 7 : Math.PI * 1.95);
    else if (sp.hat === "helmet") { cap(STEEL, back ? 0 : Math.PI * .98, back ? 7 : Math.PI * 2.02); if (!back) { ctx.fillStyle = STEEL; ctx.fillRect(-hs * .1, hy - hs * .1, hs * .2, hs * .5); } }
    else if (sp.hat === "hood") { ctx.fillStyle = sp.cloth; if (back) { disc(ctx, 0, hy, hs); ctx.fill(); ctx.stroke(); } else { ctx.beginPath(); ctx.arc(0, hy, hs, Math.PI * .8, Math.PI * 2.2); ctx.arc(0, hy + hs * .1, hs * .68, Math.PI * 2.2, Math.PI * .8, true); ctx.closePath(); ctx.fill(); ctx.stroke(); } }
    else if (sp.hat === "wizard") { ctx.fillStyle = sp.cloth; ctx.beginPath(); ctx.moveTo(-hs * 1.25, hy - hs * .35); ctx.lineTo(hs * .15, hy - hs * 2.5); ctx.lineTo(hs * 1.25, hy - hs * .35); ctx.closePath(); ctx.fill(); ctx.stroke();
      ctx.fillStyle = tm.main; ctx.fillRect(-hs * 1.05, hy - hs * .62, hs * 2.1, hs * .28); if (back) { ctx.fillStyle = sp.cloth; ctx.beginPath(); ctx.arc(0, hy, hs, 0, Math.PI); ctx.fill(); ctx.stroke(); } }
    if (!back) {
      ctx.fillStyle = sp.eye || "#1c1820"; const er = Math.max(.8, hs * .17);
      for (const d of [-1, 1]) { disc(ctx, d * hs * .38 + hs * .1, hy + hs * .12, er); ctx.fill(); }
    } else if (sp.eye && !sp.hat) { ctx.fillStyle = "rgba(0,0,0,.14)"; ctx.beginPath(); ctx.arc(0, hy, hs * .8, Math.PI * 1.1, Math.PI * 1.9); ctx.fill(); }
    return -top + hs * 1.6;
  }
  function mount(u, sp, tm, st) {
    const step = st.moving ? Math.sin(st.phase) : 0, bob = st.moving ? Math.abs(Math.cos(st.phase)) * u * .08 : 0;
    ctx.lineWidth = lw(u); ctx.strokeStyle = OUT; ctx.fillStyle = "#4a3222";
    for (const d of [-1, 1]) { const up = Math.max(0, d * step) * u * .18; rr(ctx, d * .4 * u - .1 * u, -.34 * u - up, .2 * u, .34 * u, .06 * u); ctx.fill(); ctx.stroke(); }
    ctx.fillStyle = sp.mount; ctx.beginPath(); ctx.ellipse(0, -.55 * u - bob, .78 * u, .42 * u, 0, 0, 7); ctx.fill(); ctx.stroke();
    ctx.fillStyle = sp.mount; disc(ctx, .72 * u, -.74 * u - bob, .3 * u); ctx.fill(); ctx.stroke();           // its head, leading
    ctx.fillStyle = "#f3e9d2"; ctx.beginPath(); ctx.moveTo(.86 * u, -.62 * u - bob); ctx.lineTo(1.12 * u, -.86 * u - bob); ctx.lineTo(.96 * u, -.56 * u - bob); ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.fillStyle = tm.main; rr(ctx, -.34 * u, -.82 * u - bob, .6 * u, .2 * u, .05 * u); ctx.fill(); ctx.stroke();    // saddle cloth
    ctx.save(); ctx.translate(-.06 * u, -.52 * u - bob); const h = human(u * .68, Object.assign({}, sp, { weapon: sp.weapon || "none" }), tm, { back: st.back, moving: false, legless: true }); ctx.restore();
    return .52 * u + bob + h;
  }
  function spirit(u, sp, tm, st) {
    const hop = Math.abs(Math.sin(st.phase * .5)) * u * .3, cy = -.6 * u - hop;
    ctx.fillStyle = sp.cloth; ctx.globalAlpha *= .3; disc(ctx, 0, cy, u * .95); ctx.fill(); ctx.globalAlpha /= .3;
    ctx.lineWidth = lw(u); ctx.strokeStyle = OUT; ctx.fillStyle = sp.cloth;
    ctx.beginPath(); ctx.moveTo(0, cy - u * .95); ctx.quadraticCurveTo(u * .75, cy - u * .1, u * .55, cy + u * .3); ctx.arc(0, cy + u * .3, u * .55, 0, Math.PI); ctx.quadraticCurveTo(-u * .75, cy - u * .1, 0, cy - u * .95); ctx.fill(); ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,.5)"; disc(ctx, -u * .18, cy - u * .1, u * .16); ctx.fill();
    ctx.fillStyle = tm.main; rr(ctx, -u * .5, cy + u * .42, u, u * .16, u * .06); ctx.fill();
    if (!st.back) { ctx.fillStyle = "#1c1820"; for (const d of [-1, 1]) { disc(ctx, d * u * .2, cy + u * .22, Math.max(.8, u * .08)); ctx.fill(); } }
    return u * 1.6 + hop;
  }
  function wings(u, col, tm, flap, span, yy) {
    ctx.fillStyle = col; ctx.strokeStyle = OUT; ctx.lineWidth = lw(u);
    for (const d of [-1, 1]) { ctx.beginPath(); ctx.moveTo(d * u * .3, yy - u * .1);
      ctx.quadraticCurveTo(d * u * span * .7, yy - u * (.75 + flap), d * u * span, yy - u * .25 * flap);
      ctx.quadraticCurveTo(d * u * span * .82, yy + u * .12, d * u * span * .62, yy + u * .08); ctx.quadraticCurveTo(d * u * span * .5, yy + u * .32, d * u * .3, yy + u * .32); ctx.closePath(); ctx.fill(); ctx.stroke(); }
  }
  function dragon(u, sp, tm, st) {
    const dir = st.back ? -1 : 1, flap = .35 + .35 * Math.sin(st.phase * 1.1);
    wings(u, sp.cloth, tm, flap, 1.75, 0);
    ctx.lineWidth = lw(u); ctx.strokeStyle = OUT; ctx.fillStyle = sp.cloth;
    ctx.beginPath(); ctx.moveTo(-u * .16, -dir * u * .5); ctx.lineTo(0, -dir * u * 1.15); ctx.lineTo(u * .16, -dir * u * .5); ctx.closePath(); ctx.fill(); ctx.stroke();   // tail
    ctx.beginPath(); ctx.ellipse(0, 0, u * .5, u * .66, 0, 0, 7); ctx.fill(); ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,.2)"; ctx.beginPath(); ctx.ellipse(-u * .14, -u * .14, u * .2, u * .32, 0, 0, 7); ctx.fill();
    ctx.fillStyle = tm.main; rr(ctx, -u * .46, dir * u * .2, u * .92, u * .16, u * .05); ctx.fill();
    ctx.fillStyle = sp.cloth; disc(ctx, 0, dir * u * .72, u * .36); ctx.fill(); ctx.stroke();
    if (!st.back) { ctx.fillStyle = "#1c1820"; for (const d of [-1, 1]) { disc(ctx, d * u * .15, u * .7, Math.max(.8, u * .07)); ctx.fill(); }
      ctx.fillStyle = "#fff"; for (const d of [-1, 1]) { ctx.beginPath(); ctx.moveTo(d * u * .1, u * .95); ctx.lineTo(d * u * .16, u * 1.12); ctx.lineTo(d * u * .22, u * .93); ctx.fill(); } }
    return u * 1.05;
  }
  function imp(u, sp, tm, st) {
    wings(u, sp.cloth, tm, .3 + .45 * Math.sin(st.phase * 1.6), 1.45, -u * .05);
    ctx.lineWidth = lw(u); ctx.strokeStyle = OUT; ctx.fillStyle = sp.cloth;
    for (const d of [-1, 1]) { ctx.beginPath(); ctx.moveTo(d * u * .2, -u * .45); ctx.lineTo(d * u * .5, -u * .95); ctx.lineTo(d * u * .5, -u * .3); ctx.closePath(); ctx.fill(); ctx.stroke(); }
    disc(ctx, 0, 0, u * .56); ctx.fill(); ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,.22)"; disc(ctx, -u * .18, -u * .2, u * .2); ctx.fill();
    ctx.fillStyle = tm.main; ctx.beginPath(); ctx.arc(0, 0, u * .56 - lw(u) / 2, Math.PI * .2, Math.PI * .8); ctx.closePath(); ctx.fill();
    if (!st.back) { ctx.fillStyle = "#fff"; for (const d of [-1, 1]) { disc(ctx, d * u * .2, -u * .05, u * .13); ctx.fill(); } ctx.fillStyle = "#1c1820"; for (const d of [-1, 1]) { disc(ctx, d * u * .2, -u * .02, u * .065); ctx.fill(); } }
    return u * .95;
  }
  function balloon(u, sp, tm) {
    ctx.lineWidth = lw(u); ctx.strokeStyle = OUT;
    ctx.beginPath(); ctx.moveTo(-u * .42, -u * .1); ctx.lineTo(-u * .2, u * .55); ctx.moveTo(u * .42, -u * .1); ctx.lineTo(u * .2, u * .55); ctx.stroke();
    ctx.fillStyle = WOOD; rr(ctx, -u * .3, u * .5, u * .6, u * .42, u * .08); ctx.fill(); ctx.stroke();
    ctx.fillStyle = tm.main; ctx.fillRect(-u * .3 + lw(u) / 2, u * .58, u * .6 - lw(u), u * .13);
    ctx.fillStyle = sp.cloth; ctx.beginPath(); ctx.ellipse(0, -u * .5, u * .72, u * .8, 0, 0, 7); ctx.fill(); ctx.stroke();
    ctx.save(); ctx.clip(); ctx.fillStyle = "rgba(0,0,0,.16)"; ctx.fillRect(-u * .22, -u * 1.4, u * .44, u * 2); ctx.restore();
    ctx.fillStyle = "rgba(255,255,255,.3)"; ctx.beginPath(); ctx.ellipse(-u * .3, -u * .8, u * .16, u * .26, .5, 0, 7); ctx.fill();
    return u * 1.35;
  }
  // buildings sit on a team-coloured footing
  function footing(s, tm) { ctx.fillStyle = tm.main; rr(ctx, -s, -s * .7, s * 2, s * 1.5, s * .28); ctx.fill(); ctx.strokeStyle = tm.dark; ctx.lineWidth = Math.max(1, s * .1); ctx.stroke(); }
  function turret(s, sp, tm, st) {
    const h = s * .75 * (sp.tall || 1), dir = st.back ? -1 : 1;
    footing(s, tm); ctx.lineWidth = lw(s); ctx.strokeStyle = OUT;
    let g = ctx.createLinearGradient(-s * .75, 0, s * .75, 0); g.addColorStop(0, "#8f897c"); g.addColorStop(.45, "#c9c3b5"); g.addColorStop(1, "#7d776b");
    ctx.fillStyle = g; ctx.beginPath(); ctx.ellipse(0, s * .2, s * .75, s * .34, 0, 0, Math.PI); ctx.lineTo(-s * .75, s * .2 - h); ctx.lineTo(s * .75, s * .2 - h); ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.fillStyle = "#dcd6c8"; ctx.beginPath(); ctx.ellipse(0, s * .2 - h, s * .75, s * .34, 0, 0, 7); ctx.fill(); ctx.stroke();
    const ty = s * .2 - h;
    if (sp.top === "barrel") { ctx.fillStyle = "#3b414d"; rr(ctx, -s * .2, dir > 0 ? ty - s * .05 : ty - s * .7, s * .4, s * .75, s * .1); ctx.fill(); ctx.stroke(); ctx.fillStyle = "#5a6170"; disc(ctx, 0, ty - s * .05, s * .34); ctx.fill(); ctx.stroke(); ctx.fillStyle = "#1c1f27"; disc(ctx, 0, ty + dir * s * .52 - s * .05, s * .12); ctx.fill(); }
    else if (sp.top === "bow") { ctx.strokeStyle = OUT; ctx.lineWidth = s * .2; ctx.beginPath(); ctx.arc(0, ty - dir * s * .2, s * .62, dir > 0 ? .25 : Math.PI + .25, dir > 0 ? Math.PI - .25 : -.25); ctx.stroke(); ctx.strokeStyle = WOOD; ctx.lineWidth = s * .11; ctx.stroke();
      ctx.strokeStyle = OUT; ctx.lineWidth = s * .16; ctx.beginPath(); ctx.moveTo(0, ty - s * .45); ctx.lineTo(0, ty + s * .5); ctx.stroke(); ctx.strokeStyle = "#e9eef6"; ctx.lineWidth = s * .08; ctx.stroke(); ctx.lineWidth = lw(s); ctx.strokeStyle = OUT; }
    else { const pulse = .5 + .5 * Math.sin(st.phase * .6); ctx.fillStyle = "#6c6f7c"; rr(ctx, -s * .12, ty - s * .5, s * .24, s * .5, 2); ctx.fill(); ctx.stroke();
      ctx.globalAlpha *= .35; ctx.fillStyle = sp.orb; disc(ctx, 0, ty - s * .62, s * (.5 + .15 * pulse)); ctx.fill(); ctx.globalAlpha /= .35;
      ctx.fillStyle = sp.orb; disc(ctx, 0, ty - s * .62, s * .3); ctx.fill(); ctx.stroke(); ctx.fillStyle = "rgba(255,255,255,.55)"; disc(ctx, -s * .09, ty - s * .7, s * .1); ctx.fill(); }
    return h + s * .8;
  }
  function hut(s, sp, tm) {
    footing(s, tm); ctx.lineWidth = lw(s); ctx.strokeStyle = OUT; ctx.lineJoin = "round";
    ctx.fillStyle = "#b99667"; rr(ctx, -s * .7, -s * .55, s * 1.4, s * .95, s * .1); ctx.fill(); ctx.stroke();
    ctx.fillStyle = "#3a2a1c"; rr(ctx, -s * .2, -s * .12, s * .4, s * .52, s * .16); ctx.fill();
    ctx.fillStyle = sp.roof; ctx.beginPath(); ctx.moveTo(-s * .95, -s * .45); ctx.lineTo(0, -s * 1.45); ctx.lineTo(s * .95, -s * .45); ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,.22)"; ctx.beginPath(); ctx.moveTo(-s * .62, -s * .55); ctx.lineTo(-s * .05, -s * 1.22); ctx.lineTo(-s * .2, -s * .55); ctx.closePath(); ctx.fill();
    return s * 1.5;
  }
  function tomb(s, sp, tm) {
    footing(s, tm); ctx.lineWidth = lw(s); ctx.strokeStyle = OUT;
    ctx.fillStyle = "#a9adb8"; ctx.beginPath(); ctx.moveTo(-s * .5, s * .35); ctx.lineTo(-s * .5, -s * .6); ctx.arc(0, -s * .6, s * .5, Math.PI, 0); ctx.lineTo(s * .5, s * .35); ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,.25)"; ctx.fillRect(-s * .36, -s * .7, s * .16, s * .9);
    ctx.fillStyle = BONE; disc(ctx, 0, -s * .5, s * .2); ctx.fill(); ctx.stroke(); ctx.fillStyle = "#222"; for (const d of [-1, 1]) { disc(ctx, d * s * .08, -s * .5, s * .05); ctx.fill(); }
    return s * 1.2;
  }
  const DRAW = { human, mount, spirit, dragon, imp, balloon, turret, hut, tomb };

  function unit(e, T, s) {
    const sp = spec(e.name, e.f & 1), tm = TEAM[e.o], fly = FLYERS[sp.k], building = sp.k === "turret" || sp.k === "hut" || sp.k === "tomb";
    const x = X(e.x), gy = Y(e.y), r = Math.max(.52, Math.min(1.05, (e.hw || .5) * 1.15)) * tile, u = r * (building ? 1.05 : .95);
    const moving = Math.abs(e.dx) + Math.abs(e.dy) > .004, back = e.o === 0;
    const flip = e.dx < -.01 ? -1 : e.dx > .01 ? 1 : e.x < meta.w / 2 ? 1 : -1;
    const age = s.t - e.t0, drop = !building && age < 5 ? (1 - age / 5) : 0;           // falls in when placed
    const lift = fly ? r * (1.5 + .1 * Math.sin(T * 5 + e.id)) : 0, oy = gy - lift - drop * drop * tile * 1.6;
    const al = (e.a == null ? 1 : e.a) * (e.f & 4 ? .62 : 1);
    ctx.globalAlpha = al * (fly ? .2 : .28); ctx.fillStyle = "#000";
    ctx.beginPath(); ctx.ellipse(x + r * .1, gy + r * .12, r * (fly ? .62 : .9) * (1 - drop * .4), r * .36 * (1 - drop * .4), 0, 0, 7); ctx.fill();
    ctx.globalAlpha = al;
    if (!fly && !building) {                                         // team ring under the feet
      ctx.fillStyle = tm.main; ctx.globalAlpha = al * .85; ctx.beginPath(); ctx.ellipse(x, gy + r * .04, r * .86, r * .42, 0, 0, 7); ctx.fill();
      ctx.globalAlpha = al; ctx.strokeStyle = tm.dark; ctx.lineWidth = 1.25; ctx.stroke();
      if (age >= 5 && age < 11) { ctx.strokeStyle = "rgba(255,255,255,.7)"; ctx.globalAlpha = al * (1 - (age - 5) / 6); ctx.lineWidth = 2; ctx.beginPath(); ctx.ellipse(x, gy + r * .04, r * (1 + (age - 5) * .16), r * (.5 + (age - 5) * .08), 0, 0, 7); ctx.stroke(); ctx.globalAlpha = al; }
    }
    ctx.save(); ctx.translate(x, oy); if (flip < 0 && !fly && !building) ctx.scale(-1, 1);
    const tall = DRAW[sp.k](u, sp, tm, { back, moving, phase: T * 11 + e.id * 1.3 });
    ctx.restore();
    const top = fly ? oy - tall : oy - tall, cy = fly ? oy : oy - tall * .5;
    if (e.hit) {                                                     // sparks where it was struck
      const rn = rng(e.id * 31 + Math.floor(s.t)); ctx.strokeStyle = "#fff7c2"; ctx.lineWidth = Math.max(1.5, tile * .09); ctx.lineCap = "round"; ctx.beginPath();
      for (let i = 0; i < 4; i++) { const a = rn() * 6.28, d0 = r * (.25 + rn() * .2), d1 = d0 + r * (.35 + rn() * .3); ctx.moveTo(x + Math.cos(a) * d0, cy + Math.sin(a) * d0); ctx.lineTo(x + Math.cos(a) * d1, cy + Math.sin(a) * d1); }
      ctx.stroke(); ctx.fillStyle = "rgba(255,255,255,.5)"; disc(ctx, x, cy, r * .3); ctx.fill();
    }
    if (e.f & 4 && e.t1 > e.t0) {                                    // deploy clock
      const k = Math.max(0, Math.min(1, (s.t - e.t0) / (e.t1 - e.t0))), cr = Math.max(5, tile * .3), ty = top - cr - 3;
      ctx.globalAlpha = 1; ctx.fillStyle = "rgba(8,10,16,.85)"; disc(ctx, x, ty, cr + 1.5); ctx.fill();
      ctx.fillStyle = "#f4f6fb"; disc(ctx, x, ty, cr); ctx.fill();
      ctx.fillStyle = tm.main; ctx.beginPath(); ctx.moveTo(x, ty); ctx.arc(x, ty, cr, -Math.PI / 2, -Math.PI / 2 + k * 6.2832); ctx.closePath(); ctx.fill();
    } else if (e.hp < 0.995) hpBar(x, top - Math.max(6, tile * .3), Math.max(tile * .95, r * 1.6), Math.max(3, Math.round(tile * .16)), e.hp, e.o);
    ctx.globalAlpha = al;
    if (s.labels && tile >= 12) {                                    // name tag at the feet
      const fs = Math.max(7, Math.round(tile * .34)), tw = fs * 1.75, ty = fly ? oy + r * .95 : gy + r * .3;
      ctx.fillStyle = "rgba(8,10,16,.82)"; rr(ctx, x - tw / 2, ty, tw, fs + 3, 3); ctx.fill();
      ctx.fillStyle = tm.lite; ctx.font = `700 ${fs}px Inter, system-ui, sans-serif`; ctx.fillText(abbr(e.name), x, ty + fs / 2 + 2);
    }
    ctx.globalAlpha = 1;
    if (s.hover != null && s.hover === e.id) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.beginPath(); ctx.ellipse(x, fly ? oy : gy + r * .04, r * 1.05, fly ? r * 1.05 : r * .55, 0, 0, 7); ctx.stroke(); }
    shown.push({ e, x, y: top, reach: Math.max(r, tile * .6), cy });
  }

  // a tower stands up off the board: you see its roof, and the wall facing you
  function tower(e, T) {
    const x = X(e.x), gy = Y(e.y), hw = (e.hw || 1.5) * tile, hh = (e.hh || e.hw || 1.5) * tile, king = e.name === "king_tower", tm = TEAM[e.o];
    const up = tile * (king ? 1.15 : .95), ty = gy - up, lwd = Math.max(1.25, tile * .07);
    ctx.fillStyle = "rgba(0,0,0,.26)"; rr(ctx, x - hw + tile * .2, gy - hh + tile * .2, hw * 2 + tile * .1, hh * 2 + tile * .12, tile * .3); ctx.fill();
    // wall facing the camera
    let g = ctx.createLinearGradient(x - hw, 0, x + hw, 0); g.addColorStop(0, "#8b8578"); g.addColorStop(.5, "#a8a294"); g.addColorStop(1, "#7a7468");
    ctx.fillStyle = g; rr(ctx, x - hw, ty + hh - tile * .3, hw * 2, up + tile * .3, tile * .18); ctx.fill(); ctx.strokeStyle = "#4f4b43"; ctx.lineWidth = lwd; ctx.stroke();
    ctx.save(); ctx.clip(); ctx.strokeStyle = "rgba(50,46,38,.4)"; ctx.lineWidth = 1; ctx.beginPath();
    const bs = tile * .34;
    for (let py = ty + hh + bs * .4, row = 0; py < gy + hh; py += bs, row++) { ctx.moveTo(x - hw, py); ctx.lineTo(x + hw, py);
      for (let px = x - hw + (row & 1 ? bs : bs * .2); px < x + hw; px += bs * 1.7) { ctx.moveTo(px, py); ctx.lineTo(px, py + bs); } }
    ctx.stroke(); ctx.restore();
    // banner on the wall
    const bw = hw * (king ? .5 : .56), bt = ty + hh + tile * .06, bhh = up * .78;
    ctx.fillStyle = tm.main; ctx.beginPath(); ctx.moveTo(x - bw / 2, bt); ctx.lineTo(x + bw / 2, bt); ctx.lineTo(x + bw / 2, bt + bhh); ctx.lineTo(x, bt + bhh * .78); ctx.lineTo(x - bw / 2, bt + bhh); ctx.closePath(); ctx.fill();
    ctx.strokeStyle = tm.dark; ctx.lineWidth = lwd; ctx.stroke();
    ctx.fillStyle = tm.lite; disc(ctx, x, bt + bhh * .36, bw * .2); ctx.fill();
    // roof: stone rim with battlements, team deck inside
    g = ctx.createLinearGradient(0, ty - hh, 0, ty + hh); g.addColorStop(0, "#e2dccd"); g.addColorStop(1, "#bdb7a8");
    ctx.fillStyle = g; rr(ctx, x - hw, ty - hh, hw * 2, hh * 2, tile * .26); ctx.fill(); ctx.strokeStyle = "#5f5a51"; ctx.lineWidth = lwd; ctx.stroke();
    const n = king ? 4 : 3, m = (hw * 2) / (n * 2 - 1);
    for (let i = 0; i < n; i++) for (const yy of [ty - hh - m * .22, ty + hh - m * .5]) {
      ctx.fillStyle = "#efe9db"; rr(ctx, x - hw + i * 2 * m, yy, m, m * .72, 2); ctx.fill(); ctx.strokeStyle = "#6c675d"; ctx.lineWidth = 1; ctx.stroke(); }
    const ix = hw * .72, iy = hh * .66;
    g = ctx.createLinearGradient(0, ty - iy, 0, ty + iy); g.addColorStop(0, tm.lite); g.addColorStop(1, tm.main);
    ctx.fillStyle = g; rr(ctx, x - ix, ty - iy, ix * 2, iy * 2, tile * .2); ctx.fill(); ctx.strokeStyle = tm.dark; ctx.lineWidth = lwd; ctx.stroke();
    if (king) {
      const dir = e.o === 0 ? -1 : 1;                                // a cannon under the crown
      ctx.fillStyle = "#39404d"; rr(ctx, x - hw * .13, dir < 0 ? ty - hh * .78 : ty + hh * .1, hw * .26, hh * .68, 3); ctx.fill(); ctx.strokeStyle = OUT; ctx.lineWidth = 1; ctx.stroke();
      crown(x, ty - hh * .02, hw * .98);
    } else {                                                         // an archer on the roof
      ctx.save(); ctx.translate(x, ty + iy * .62);
      human(hw * .46, { cloth: tm.dark, hat: "hood", weapon: "bow", skin: SKIN }, tm, { back: e.o === 0, moving: false, legless: true, phase: 0 });
      ctx.restore();
    }
    if (e.hit) { ctx.fillStyle = "rgba(255,255,255,.3)"; rr(ctx, x - hw, ty - hh, hw * 2, hh * 2 + up, tile * .26); ctx.fill(); }
    const bh = Math.max(4, Math.round(tile * .26));
    hpBar(x, e.o === 0 ? gy + hh + tile * .22 : ty - hh - tile * .34 - bh, hw * 1.7, bh, e.hp, e.o);
    shown.push({ e, x, y: ty - hh, reach: Math.max(hw, hh) + up * .5, cy: gy - up * .5 });
  }
  function rubble(d) {
    const x = X(d.x), y = Y(d.y), hw = (d.hw || 1.5) * tile, hh = (d.hh || d.hw || 1.5) * tile, rn = rng(Math.round(d.x * 7 + d.y * 13) + 3);
    ctx.fillStyle = "rgba(20,16,10,.42)"; rr(ctx, x - hw, y - hh, hw * 2, hh * 2, tile * .5); ctx.fill();
    ctx.fillStyle = "#8f897c"; rr(ctx, x - hw * .9, y + hh * .45, hw * 1.8, hh * .5, 3); ctx.fill(); ctx.strokeStyle = "rgba(0,0,0,.45)"; ctx.lineWidth = 1; ctx.stroke();
    for (let i = 0; i < 12; i++) {
      const px = x + (rn() - .5) * hw * 1.7, py = y + (rn() - .5) * hh * 1.5, sz = tile * (.22 + rn() * .34);
      ctx.fillStyle = i % 3 ? "#8d887c" : "#b9b3a6"; rr(ctx, px - sz / 2, py - sz / 2, sz, sz * .78, 2); ctx.fill(); ctx.stroke();
    }
  }

  // ---------- spell effects ----------
  function spell(sp, ev, age, T) {
    const tps = meta.tps, dur = sp[2] * tps, k = age / dur, R = sp[0] * tile, col = sp[1], style = sp[3], x = X(ev[3]), y = Y(ev[4]);
    const rn = rng(ev[0] * 13 + ev[1] + 1), fade = sp[2] < 1.5 ? 1 - k : Math.min(1, (1 - k) * 6) * Math.min(1, age / 4);
    const glow = (r, a) => { const g = ctx.createRadialGradient(x, y, 0, x, y, r); g.addColorStop(0, col); g.addColorStop(1, "rgba(0,0,0,0)"); ctx.globalAlpha = a; ctx.fillStyle = g; disc(ctx, x, y, r); ctx.fill(); };
    const ring = (a) => { ctx.globalAlpha = a; ctx.strokeStyle = col; ctx.lineWidth = 2; disc(ctx, x, y, R); ctx.stroke(); };
    if (style === "roll") {                                          // a log rolling up the lane
      const dir = ev[1] === 0 ? 1 : -1, ly = Y(ev[4] + dir * sp[4] * Math.min(1, k)), hw = R, hh = tile * .42;
      ctx.globalAlpha = .3 * (1 - k); ctx.fillStyle = "#6b4a22"; ctx.fillRect(x - hw, Math.min(y, ly), hw * 2, Math.abs(ly - y));
      ctx.globalAlpha = Math.min(1, (1 - k) * 5); ctx.fillStyle = col; rr(ctx, x - hw, ly - hh, hw * 2, hh * 2, hh); ctx.fill(); ctx.strokeStyle = OUT; ctx.lineWidth = 1.5; ctx.stroke();
      ctx.strokeStyle = "rgba(60,35,12,.6)"; ctx.lineWidth = 1; ctx.beginPath(); const ph = (T * 9) % 1;
      for (let i = 0; i < 3; i++) { const yy = ly - hh + ((i + ph) / 3) * hh * 2; ctx.moveTo(x - hw + hh * .5, yy); ctx.lineTo(x + hw - hh * .5, yy); } ctx.stroke();
      ctx.globalAlpha = 1; return;
    }
    if (style === "fire") {
      const grow = .35 + .65 * Math.min(1, k * 4);
      glow(R * grow * 1.15, .9 * (1 - k)); ctx.globalAlpha = .85 * (1 - k) * (1 - k); ctx.fillStyle = "#fff6d0"; disc(ctx, x, y, R * grow * .45); ctx.fill();
      ctx.fillStyle = "#ffd36b"; ctx.globalAlpha = 1 - k;
      for (let i = 0; i < 10; i++) { const a = rn() * 6.28, d = R * (.3 + rn() * .9) * Math.min(1, k * 2.2); disc(ctx, x + Math.cos(a) * d, y + Math.sin(a) * d - k * tile * .6, Math.max(1, tile * .1 * (1 - k))); ctx.fill(); }
      ring(.7 * (1 - k));
    } else if (style === "bolt") {
      glow(R, .55 * (1 - k)); ring(.8 * (1 - k));
      if (k < .7) { ctx.globalAlpha = 1 - k; ctx.lineJoin = "miter";
        for (let b = 0; b < 3; b++) { const a = rn() * 6.28, d = R * rn() * .7, bx = x + Math.cos(a) * d, by = y + Math.sin(a) * d; ctx.beginPath(); let px = bx + (rn() - .5) * tile, py = by - tile * 5; ctx.moveTo(px, py);
          for (let i = 1; i <= 5; i++) { px = bx + (rn() - .5) * tile * (1 - i / 5) * 1.6; py = by - tile * 5 * (1 - i / 5); ctx.lineTo(px, py); }
          ctx.strokeStyle = col; ctx.lineWidth = Math.max(3, tile * .2); ctx.stroke(); ctx.strokeStyle = "#fff"; ctx.lineWidth = Math.max(1.25, tile * .07); ctx.stroke(); } }
    } else if (style === "arrows") {
      ctx.globalAlpha = .16 * (1 - k); ctx.fillStyle = col; disc(ctx, x, y, R); ctx.fill(); ring(.6 * (1 - k));
      ctx.lineCap = "round";
      for (let i = 0; i < 26; i++) { const a = rn() * 6.28, d = R * Math.sqrt(rn()), ax = x + Math.cos(a) * d, ay = y + Math.sin(a) * d, land = .15 + rn() * .3, fall = Math.max(0, 1 - k / land), len = tile * .5;
        ctx.globalAlpha = Math.min(1, (1 - k) * 2); ctx.strokeStyle = "#5b3d1e"; ctx.lineWidth = Math.max(1.25, tile * .07); ctx.beginPath(); ctx.moveTo(ax - fall * tile * 2.5, ay - fall * tile * 5); ctx.lineTo(ax - fall * tile * 2.5 - len * .45, ay - fall * tile * 5 - len); ctx.stroke();
        ctx.strokeStyle = "#f2efe6"; ctx.beginPath(); ctx.moveTo(ax - fall * tile * 2.5 - len * .36, ay - fall * tile * 5 - len * .8); ctx.lineTo(ax - fall * tile * 2.5 - len * .45, ay - fall * tile * 5 - len); ctx.stroke(); }
    } else if (style === "swirl") {
      glow(R, .3 * fade); ctx.lineCap = "round";
      for (let i = 0; i < 5; i++) { const rr2 = R * (.2 + i * .18), a0 = T * (9 - i) + i * 1.3; ctx.globalAlpha = .75 * fade; ctx.strokeStyle = i & 1 ? "#fff" : col; ctx.lineWidth = Math.max(2, tile * .14); ctx.beginPath(); ctx.arc(x, y, rr2, a0, a0 + 2.2); ctx.stroke(); }
    } else {
      ctx.globalAlpha = .3 * fade; ctx.fillStyle = col; disc(ctx, x, y, R); ctx.fill(); glow(R * .9, .35 * fade); ring(.85 * fade);
      if (style === "cloud") { ctx.fillStyle = col;
        for (let i = 0; i < 9; i++) { const a = rn() * 6.28, d = R * Math.sqrt(rn()) * .85, ph = (T * (.5 + rn() * .5) + rn()) % 1; ctx.globalAlpha = .75 * fade * (1 - ph); disc(ctx, x + Math.cos(a) * d, y + Math.sin(a) * d - ph * tile * 1.1, tile * (.1 + .12 * ph)); ctx.fill(); } }
      else if (style === "ice") { ctx.strokeStyle = "#fff"; ctx.lineWidth = Math.max(1, tile * .06); ctx.globalAlpha = .85 * fade;
        for (let i = 0; i < 7; i++) { const a = rn() * 6.28, d = R * Math.sqrt(rn()) * .8, fx = x + Math.cos(a) * d, fy = y + Math.sin(a) * d, s = tile * (.16 + rn() * .14); ctx.beginPath();
          for (let j = 0; j < 3; j++) { const b = j * Math.PI / 3; ctx.moveTo(fx - Math.cos(b) * s, fy - Math.sin(b) * s); ctx.lineTo(fx + Math.cos(b) * s, fy + Math.sin(b) * s); } ctx.stroke(); } }
      else if (style === "quake") { ctx.strokeStyle = "rgba(60,40,20,.8)"; ctx.lineWidth = Math.max(1.5, tile * .09); ctx.globalAlpha = fade; ctx.lineJoin = "round";
        for (let i = 0; i < 6; i++) { const a = rn() * 6.28; ctx.beginPath(); let d = R * .1; ctx.moveTo(x + Math.cos(a) * d, y + Math.sin(a) * d);
          for (let j = 0; j < 4; j++) { d += R * .2; const aa = a + (rn() - .5) * .6; ctx.lineTo(x + Math.cos(aa) * d, y + Math.sin(aa) * d); } ctx.stroke(); } }
    }
    ctx.globalAlpha = 1;
  }

  function draw(s) {
    if (!bg) return;
    const tps = meta.tps, T = s.t / tps;
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.drawImage(bg, 0, 0); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.lineCap = "round";
    // light on the water, drifting with match time (so a paused frame is still)
    const half = meta.river_half || 1, bh = meta.bridge_half || 1.5;
    ctx.lineWidth = Math.max(1.25, tile * .08);
    for (let i = 0; i < 12; i++) {
      const span = meta.w + 2, raw = i * 1.83 + T * (i % 2 ? .34 : -.26), wx = ((raw % span) + span) % span - 1, len = .5 + (i % 3) * .25;
      if (meta.bridges.some((bx) => Math.abs(wx + len / 2 - bx) < bh + len / 2 + .2) || wx < .1 || wx > meta.w - len - .1) continue;
      const wy = meta.river_y + (((i * .37) % 1) * 1.3 - .65) * half, sway = Math.sin(T * 1.5 + i) * tile * .05;
      ctx.strokeStyle = `rgba(255,255,255,${.22 + .16 * Math.sin(T * 2 + i * 1.7)})`;
      ctx.beginPath(); ctx.moveTo(X(wx), Y(wy) + sway); ctx.quadraticCurveTo(X(wx + len / 2), Y(wy) - tile * .1 + sway, X(wx + len), Y(wy) + sway); ctx.stroke();
    }
    // where a tower fell
    for (const d of s.deaths || []) if (d.tower && d.t <= s.t) rubble(d);
    // casts: an area effect for spells, a ripple where a troop was placed
    for (const ev of s.events || []) {
      const age = s.t - ev[0];
      if (age < 0 || ev[2] === "ability") continue;
      const sp = SPELL[ev[2]];
      if (sp) { if (age <= sp[2] * tps) spell(sp, ev, age, T); }
      else if (age <= 16) {
        const k = age / 16; ctx.strokeStyle = TEAM[ev[1]].lite; ctx.globalAlpha = 1 - k; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.ellipse(X(ev[3]), Y(ev[4]), tile * (.5 + 1.6 * k), tile * (.3 + .9 * k), 0, 0, 7); ctx.stroke(); ctx.globalAlpha = 1;
      }
    }
    // everything on the ground from the back of the board to the front, then flyers over the top
    shown = [];
    const front = (e) => (e.f & 2 ? Y(e.y) + (e.hh || e.hw || 1.5) * tile : Y(e.y));
    const ents = s.ents.slice().sort((a, b) => ((a.f & 3) === 1) - ((b.f & 3) === 1) || front(a) - front(b));
    for (const e of ents) { if (e.f & 2) tower(e, T); else unit(e, T, s); }
    // a puff where something just died; a blast and a crown where a tower just fell
    for (const d of s.deaths || []) {
      const age = s.t - d.t, len = d.tower ? 44 : 10;
      if (age < 0 || age > len) continue;
      const x = X(d.x), y = Y(d.y), rn = rng(Math.round(d.x * 17 + d.y * 29 + d.t));
      if (d.tower) {
        const k = Math.min(1, age / 22), R = (d.hw || 1.5) * tile * (1 + 1.5 * k);
        if (age <= 22) {
          ctx.globalAlpha = .85 * (1 - k); ctx.fillStyle = "#ffb347"; disc(ctx, x, y, R); ctx.fill();
          ctx.fillStyle = "#fff4c2"; disc(ctx, x, y, R * .55); ctx.fill();
          ctx.strokeStyle = "#fff"; ctx.lineWidth = 3; disc(ctx, x, y, R * 1.15); ctx.stroke();
          ctx.fillStyle = "#a59f92"; ctx.strokeStyle = "rgba(0,0,0,.5)"; ctx.lineWidth = 1; ctx.globalAlpha = 1 - k * k;
          for (let i = 0; i < 10; i++) { const a = rn() * 6.28, dist = R * (.4 + rn() * .9), sz = tile * (.18 + rn() * .2); rr(ctx, x + Math.cos(a) * dist - sz / 2, y + Math.sin(a) * dist - sz / 2 - Math.sin(k * Math.PI) * tile * 1.4, sz, sz, 2); ctx.fill(); ctx.stroke(); }
        }
        const kc = age / len; ctx.globalAlpha = Math.min(1, (1 - kc) * 3) * Math.min(1, age / 5);
        crown(x, y - tile * (.6 + 2 * kc), tile * (1.2 + .5 * Math.min(1, age / 8)));
      } else {
        const k = age / len, R = (d.r || .5) * tile;
        ctx.fillStyle = "#f6f1ff";
        for (let i = 0; i < 6; i++) { const a = rn() * 6.28, dist = R * (.3 + 1.2 * k) * (.5 + rn() * .6); ctx.globalAlpha = .7 * (1 - k); disc(ctx, x + Math.cos(a) * dist, y - R * .5 + Math.sin(a) * dist * .7 - k * tile * .3, R * (.5 + rn() * .35) * (1 - k * .5)); ctx.fill(); }
        ctx.globalAlpha = 1 - k; ctx.fillStyle = "#e36bd8"; const dy = y - R * .8 - k * tile * 1.1, ds = Math.max(2, tile * .16);      // a drop of elixir
        ctx.beginPath(); ctx.moveTo(x, dy - ds * 1.5); ctx.quadraticCurveTo(x + ds * 1.2, dy + ds * .2, x, dy + ds); ctx.quadraticCurveTo(x - ds * 1.2, dy + ds * .2, x, dy - ds * 1.5); ctx.fill();
      }
      ctx.globalAlpha = 1;
    }
  }
  // the drawn thing under a point (canvas CSS pixels)
  function pick(mx, my) {
    let best = null, bd = 1e9;
    for (const h of shown) { const d = Math.hypot(h.x - mx, h.cy - my); if (d < h.reach && d < bd) { bd = d; best = h; } }
    return best;
  }
  function point(mx, my) {
    const x = (mx - pad) / tile, y = meta.h - (my - pad) / tile;
    return x >= 0 && x < meta.w && y >= 0 && y < meta.h ? { x, y } : null;
  }
  function target(p) {
    if (!p) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "rgba(255,184,41,.35)"; ctx.strokeStyle = "#ffb829"; ctx.lineWidth = 2;
    ctx.fillRect(X(Math.floor(p.x)), Y(Math.floor(p.y) + 1), tile, tile);
    ctx.strokeRect(X(Math.floor(p.x)), Y(Math.floor(p.y) + 1), tile, tile);
  }
  return { configure, resize, draw, pick, point, target, team: TEAM };
}
// ===== end arena board renderer =============================================
