// Smooth scroll for internal anchors
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener("click", e => {
      const t = document.querySelector(a.getAttribute("href"));
      if (!t) return;
      e.preventDefault();
      t.scrollIntoView({ behavior: "smooth" });
    });
  });

  // cart badge from localStorage (future)
  const badge = document.getElementById("cart-count");
  if (badge) {
    const n = Number(localStorage.getItem("cart_count") || 0);
    badge.textContent = isNaN(n) ? "0" : String(n);
  }
});
