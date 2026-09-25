/**
 * LoanLens — Global App Controller
 * Theme switching, mobile drawer navigation, keyboard dismiss, and smooth interactions.
 */

(function () {
  'use strict';

  // 1. Mobile Drawer Navigation
  const sidebar = document.getElementById('appSidebar');
  const backdrop = document.getElementById('drawerBackdrop');
  const menuBtn = document.getElementById('mobileMenuBtn');
  const closeBtn = document.getElementById('drawerCloseBtn');

  function openDrawer() {
    if (!sidebar || !backdrop) return;
    sidebar.classList.add('drawer-open');
    backdrop.classList.add('active');
    if (menuBtn) menuBtn.setAttribute('aria-expanded', 'true');
    document.body.style.overflow = 'hidden';
  }

  function closeDrawer() {
    if (!sidebar || !backdrop) return;
    sidebar.classList.remove('drawer-open');
    backdrop.classList.remove('active');
    if (menuBtn) menuBtn.setAttribute('aria-expanded', 'false');
    document.body.style.overflow = '';
  }

  if (menuBtn) menuBtn.addEventListener('click', openDrawer);
  if (closeBtn) closeBtn.addEventListener('click', closeDrawer);
  if (backdrop) backdrop.addEventListener('click', closeDrawer);

  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && sidebar && sidebar.classList.contains('drawer-open')) {
      closeDrawer();
    }
  });

  // 2. Theme Toggle Controller
  const themeBtn = document.getElementById('themeToggleBtn');
  if (themeBtn) {
    themeBtn.addEventListener('click', function () {
      const current = document.documentElement.getAttribute('data-theme') || 'light';
      const next = current === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('loanlens-theme', next);
    });
  }
})();
