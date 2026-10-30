/* Figure 1 GIF playback and the article's left navigation. */
(function () {
  'use strict';
  const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
  const hero = document.querySelector('[data-hero-animation]');
  if (hero) {
    const button = hero.querySelector('[data-hero-toggle]');
    const img = hero.querySelector('picture img');
    const sources = Array.from(hero.querySelectorAll('picture source'));
    let playing = false, restoreAfterPrint = false;
    function show(motion) {
      playing = motion;
      sources.forEach(source => { source.srcset = motion ? source.dataset.motion : source.dataset.poster; });
      img.src = motion ? img.dataset.motion : img.dataset.poster;
      button.textContent = motion ? '暂停 · 显示全部数据' : '播放动画';
      button.setAttribute('aria-pressed', String(motion));
    }
    button.hidden = false;
    button.addEventListener('click', () => show(!playing));
    if (preference.addEventListener) preference.addEventListener('change', e => { if (e.matches) show(false); });
    window.addEventListener('beforeprint', () => { restoreAfterPrint = playing; show(false); });
    window.addEventListener('afterprint', () => { if (restoreAfterPrint && !preference.matches) show(true); });
    document.addEventListener('visibilitychange', () => { if (document.hidden) show(false); });
    show(!preference.matches);
  }
  const nav = document.querySelector('.article-toc');
  if (nav && 'IntersectionObserver' in window) {
    const links = Array.from(nav.querySelectorAll('a[href^="#"]'));
    const sections = links.map(a => document.getElementById(a.getAttribute('href').slice(1))).filter(Boolean);
    const visible = new Map();
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => visible.set(entry.target.id, entry.isIntersecting));
      const target = sections.find(section => visible.get(section.id));
      if (!target) return;
      links.forEach(a => {
        if (a.getAttribute('href') === '#' + target.id) a.setAttribute('aria-current', 'location');
        else a.removeAttribute('aria-current');
      });
    }, { rootMargin: '0px 0px -55% 0px', threshold: 0 });
    sections.filter(section => section.id !== 'top').forEach(section => observer.observe(section));
  }
})();
