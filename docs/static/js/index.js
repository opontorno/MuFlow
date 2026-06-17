// ── Navbar burger (mobile) ─────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const burger = document.querySelector('.navbar-burger');
  const menu   = document.querySelector('.navbar-menu');
  if (burger) {
    burger.addEventListener('click', () => {
      burger.classList.toggle('is-active');
      menu.classList.toggle('is-active');
    });
  }

  // Close menu on nav-link click (mobile)
  document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', () => {
      burger.classList.remove('is-active');
      menu.classList.remove('is-active');
    });
  });
});

// ── BibTeX copy ────────────────────────────────────────────────────────────
function copyBibtex() {
  const text = document.getElementById('bibtex-content').innerText;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById('copy-btn');
    btn.innerHTML = '<i class="fas fa-check"></i>&nbsp;Copied!';
    btn.style.background = '#40a070';
    setTimeout(() => {
      btn.innerHTML = '<i class="far fa-copy"></i>&nbsp;Copy';
      btn.style.background = '';
    }, 2000);
  });
}

// ── Active navbar link on scroll ──────────────────────────────────────────
const sections = document.querySelectorAll('section[id]');
const navLinks  = document.querySelectorAll('.nav-link');

window.addEventListener('scroll', () => {
  let current = '';
  sections.forEach(sec => {
    if (window.scrollY >= sec.offsetTop - 80) current = sec.id;
  });
  navLinks.forEach(link => {
    link.classList.remove('is-active-nav');
    if (link.getAttribute('href') === `#${current}`) {
      link.classList.add('is-active-nav');
    }
  });
}, { passive: true });
