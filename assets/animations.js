/* Deterministic toy dynamics and a guided reveal of unchanged experimental panels. */
(function () {
  'use strict';
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const controllers = [];
  function player(root, max, duration, render, label) {
    const controls = root.querySelector('.animation-controls');
    const play = controls.querySelector('[data-action="play"]');
    const replay = controls.querySelector('[data-action="replay"]');
    const slider = controls.querySelector('[data-action="scrub"]');
    let value = Number(slider.value), running = false, frame = 0, previous = 0;
    controls.hidden = false;
    const stoppedLabel = label;
    function draw() { slider.value = Math.round(value); render(value); }
    function stop() { running = false; cancelAnimationFrame(frame); play.textContent = stoppedLabel; play.setAttribute('aria-pressed', 'false'); }
    function tick(now) {
      if (!running) return;
      if (previous) value = Math.min(max, value + (now - previous) * max / duration);
      previous = now; draw();
      if (value >= max) stop(); else frame = requestAnimationFrame(tick);
    }
    function start() {
      if (value >= max) value = 0;
      running = true; previous = 0; play.textContent = 'Pause'; play.setAttribute('aria-pressed', 'true'); draw(); frame = requestAnimationFrame(tick);
    }
    play.addEventListener('click', () => running ? stop() : start());
    replay.addEventListener('click', () => { stop(); value = 0; draw(); if (!reduceMotion) start(); });
    slider.addEventListener('input', () => { stop(); value = Number(slider.value); draw(); });
    play.setAttribute('aria-pressed', 'false');
    draw(); controllers.push({ stop });
    if ('IntersectionObserver' in window) {
      const observer = new IntersectionObserver(entries => { if (!entries[0].isIntersecting) stop(); }, { threshold: 0 });
      observer.observe(root);
    }
  }
  document.querySelectorAll('[data-edge-demo]').forEach(root => {
    const panels = Array.from(root.querySelectorAll('.quadratic-panel'));
    const stepText = root.querySelector('[data-step]');
    const point = x => ({ x: 120 + x * 31.25, y: 138 - x * x * (118 / 10.24) });
    player(root, 6, 7200, value => {
      const step = Math.floor(value + 1e-8);
      panels.forEach(panel => {
        const ratio = Number(panel.dataset.ratio), x = Math.pow(1 - ratio, step);
        const p = point(x), dot = panel.querySelector('.iterate'), trail = panel.querySelector('.step-trail');
        dot.setAttribute('cx', p.x.toFixed(3)); dot.setAttribute('cy', p.y.toFixed(3));
        let d = '';
        for (let k = 0; k <= step; k++) { const q = point(Math.pow(1 - ratio, k)); d += (k ? ' L ' : 'M ') + q.x.toFixed(3) + ' ' + q.y.toFixed(3); }
        trail.setAttribute('d', d); trail.style.stroke = panel.dataset.color;
        panel.querySelector('output').textContent = 'x' + step + ' = ' + x.toFixed(3) + ' · L/L₀ = ' + (x * x).toFixed(3);
      });
      stepText.textContent = step + ' / 6';
    }, 'Play');
  });
  // Preserve ordinary image URLs while providing an accessible in-page viewer.
  if (typeof HTMLDialogElement !== 'undefined') {
    document.addEventListener('click', event => {
      const link = event.target.closest && event.target.closest('a.figure-link');
      if (!link || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      const dialog = document.createElement('dialog');
      dialog.className = 'figure-viewer'; dialog.setAttribute('aria-label', 'Full-size figure');
      const close = document.createElement('button'); close.type = 'button'; close.textContent = 'Close figure';
      const img = document.createElement('img'); img.src = link.href;
      img.alt = link.querySelector('img').alt;
      dialog.append(close, img); document.body.append(dialog);
      close.addEventListener('click', () => dialog.close());
      dialog.addEventListener('click', e => { if (e.target === dialog) dialog.close(); });
      dialog.addEventListener('close', () => { dialog.remove(); link.focus(); });
      dialog.showModal(); close.focus();
    });
  }
  document.addEventListener('visibilitychange', () => { if (document.hidden) controllers.forEach(c => c.stop()); });
})();
(function () {
  'use strict';
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  document.querySelectorAll('[data-geometry-demo]').forEach(root => {
    const controls=root.querySelector('.geometry-controls'), slider=root.querySelector('[data-g-progress]');
    const play=root.querySelector('[data-g-play]'), reset=root.querySelector('[data-g-reset]');
    const panel=root.querySelector('[data-geometry-panel="2"]');
    let value=100, running=false, frame=0, last=0;
    function render(){
      const strength=Math.pow(4,1-value/100), r=strength*23;
      panel.querySelector('.mapped-circle').setAttribute('rx',r.toFixed(3));
      panel.querySelector('.singular-one').setAttribute('d','M0,0 L'+r.toFixed(3)+',0');
      panel.querySelector('.bar-one').style.width=(strength/4*100)+'%';
      panel.querySelector('.value-one').textContent=strength.toFixed(2);
      panel.querySelector('p').textContent=value>=99.99?'Active strengths become 1 : 1':'Illustrated transform: '+strength.toFixed(2)+' : 1';
      slider.value=Math.round(value); root.querySelector('[data-g-label]').textContent=Math.round(value)+'%';
    }
    function stop(){running=false;cancelAnimationFrame(frame);play.textContent='Play transform';play.setAttribute('aria-pressed','false');}
    function tick(now){if(!running)return;if(last)value=Math.min(100,value+(now-last)/45);last=now;render();if(value>=100)stop();else frame=requestAnimationFrame(tick);}
    function start(){if(value>=100)value=0;running=true;last=0;play.textContent='Pause';play.setAttribute('aria-pressed','true');render();frame=requestAnimationFrame(tick);}
    play.addEventListener('click',()=>running?stop():start());
    reset.addEventListener('click',()=>{stop();value=0;render();});
    slider.addEventListener('input',()=>{stop();value=Number(slider.value);render();});
    document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
    if('IntersectionObserver' in window)new IntersectionObserver(es=>{if(!es[0].isIntersecting)stop();}).observe(root);
    controls.hidden=false; render();
    if(reduced)root.setAttribute('data-reduced-motion','true');
  });
  window.addEventListener('beforeprint',()=>{
    document.querySelectorAll('details.technical').forEach(d=>{d.dataset.wasOpen=d.open?'1':'0';d.open=true;});
  });
  window.addEventListener('afterprint',()=>{
    document.querySelectorAll('details.technical').forEach(d=>{d.open=d.dataset.wasOpen==='1';delete d.dataset.wasOpen;});
  });
})();
