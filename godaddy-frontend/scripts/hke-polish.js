(function () {
  function ready(fn) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", fn, { once: true });
    } else {
      fn();
    }
  }

  function createHomeLink(extraClass, text) {
    var link = document.createElement("a");
    link.href = "index.html";
    link.textContent = text || "Home";
    link.className = (extraClass ? extraClass + " " : "") + "hke-home-link";
    return link;
  }

  function hasHomeLink(root) {
    return !!(root && root.querySelector('a[href="index.html"], a[href="./index.html"], a[href="/"], a[aria-label="Home"]'));
  }

  function applyBasicPageShell() {
    var body = document.body;
    if (!body) return;

    var hasStructuredHeader = document.querySelector(".nav-wrap, .topbar, .portal-topbar, nav.navbar, .top");
    if (hasStructuredHeader) return;

    body.classList.add("hke-basic-page");

    if (!document.querySelector(".hke-polish-topbar")) {
      var topbar = document.createElement("header");
      topbar.className = "hke-polish-topbar";
      topbar.innerHTML = [
        '<div class="inner">',
        '  <a class="hke-polish-brand" href="index.html" aria-label="Home">',
        '    <img src="media/logo.png" alt="Himalayan Kerala Expeditions Logo" width="550" height="550" loading="eager" decoding="async">',
        '    <div>',
        '      <p class="line1">Himalayan Kerala Expeditions</p>',
        '      <p class="line2">Premium travel planning</p>',
        '    </div>',
        "  </a>",
        '  <div class="hke-polish-actions"></div>',
        "</div>"
      ].join("");

      topbar.querySelector(".hke-polish-actions").appendChild(createHomeLink("", "Home"));
      body.insertBefore(topbar, body.firstChild);
    }
  }

  function ensureHomeLink() {
    var navList = document.querySelector(".navbar-nav");
    if (navList && !hasHomeLink(navList)) {
      var li = document.createElement("li");
      li.className = "nav-item";
      var link = document.createElement("a");
      link.className = "nav-link";
      link.href = "index.html";
      link.textContent = "Home";
      li.appendChild(link);
      navList.insertBefore(li, navList.firstChild);
      return;
    }

    var portalActions = document.querySelector(".portal-topbar-actions");
    if (portalActions && !hasHomeLink(portalActions)) {
      var portalLink = document.createElement("a");
      portalLink.href = "index.html";
      portalLink.textContent = "Home";
      portalLink.className = "portal-btn portal-btn-ghost hke-home-link";
      portalActions.insertBefore(portalLink, portalActions.firstChild);
      return;
    }

    var topActions = document.querySelector(".top-actions");
    if (topActions && !hasHomeLink(topActions)) {
      topActions.insertBefore(createHomeLink("", "Home"), topActions.firstChild);
      return;
    }

    var buttonRow = document.querySelector(".topbar .container .d-flex:last-child, .topbar .container .btn-wrap, .topbar .container .actions");
    if (buttonRow && !hasHomeLink(buttonRow)) {
      var buttonLink = buttonRow.querySelector(".btn-gold") ? createHomeLink("btn-ghost", "Home") : createHomeLink("", "Home");
      buttonRow.insertBefore(buttonLink, buttonRow.firstChild);
      return;
    }

    var genericTopbar = document.querySelector(".topbar .container, .nav-wrap .container, nav.navbar .container");
    if (genericTopbar && !hasHomeLink(genericTopbar)) {
      var actionWrap = document.createElement("div");
      actionWrap.className = "hke-polish-actions";
      actionWrap.appendChild(createHomeLink("", "Home"));
      genericTopbar.appendChild(actionWrap);
    }
  }

  function ensureFooter() {
    var footer = document.querySelector("footer");
    if (!footer) {
      footer = document.createElement("footer");
      document.body.appendChild(footer);
    }

    footer.classList.add("hke-shared-footer");
    footer.innerHTML = [
      '<div class="container">',
      '  <div class="hke-footer-copy">&copy; Himalayan Kerala Expeditions Estd 2025.</div>',
      '  <div class="hke-footer-meta">Email: <strong>info@himalayankeralaexpeditions.com</strong> <span style="opacity:.5;">|</span> WhatsApp: <strong>+91 97972 94747</strong></div>',
      '  <div class="hke-footer-links">',
      '    <a href="terms.html">Terms &amp; Conditions</a>',
      '    <span style="opacity:.5;">|</span>',
      '    <a href="privacy.html">Privacy Policy</a>',
      "  </div>",
      "</div>"
    ].join("");
  }

  function ensureImageAttributes() {
    var sizeMap = {
      "media/logo.png": { width: 550, height: 550 },
      "media/gallery1.jpg": { width: 1536, height: 1024 },
      "media/whatsapp-icon.png": { width: 1024, height: 1024 }
    };

    document.querySelectorAll("img").forEach(function (img, index) {
      if (!img.hasAttribute("decoding")) {
        img.setAttribute("decoding", "async");
      }

      if (!img.hasAttribute("loading")) {
        var eager = index === 0 && img.closest("header, .hero, .nav-wrap, .topbar, .portal-topbar");
        img.setAttribute("loading", eager ? "eager" : "lazy");
      }

      var src = img.getAttribute("src") || "";
      var dims = sizeMap[src];
      if (dims) {
        if (!img.getAttribute("width")) img.setAttribute("width", String(dims.width));
        if (!img.getAttribute("height")) img.setAttribute("height", String(dims.height));
      }
    });
  }

  function bindNavbarFallback() {
    var bootstrapAvailable = !!(window.bootstrap && window.bootstrap.Collapse);

    document.querySelectorAll(".navbar-toggler[data-bs-target]").forEach(function (btn) {
      var targetSelector = btn.getAttribute("data-bs-target");
      var target = targetSelector ? document.querySelector(targetSelector) : null;
      if (!target) return;

      if (!bootstrapAvailable) {
        btn.addEventListener("click", function () {
          var isOpen = target.classList.contains("show");
          target.classList.toggle("show", !isOpen);
          btn.setAttribute("aria-expanded", isOpen ? "false" : "true");
        });
      }

      target.querySelectorAll("a").forEach(function (link) {
        link.addEventListener("click", function () {
          if (target.classList.contains("show")) {
            target.classList.remove("show");
            btn.setAttribute("aria-expanded", "false");
          }
        });
      });
    });
  }

  ready(function () {
    applyBasicPageShell();
    ensureHomeLink();
    ensureFooter();
    ensureImageAttributes();
    bindNavbarFallback();
  });
})();
