(function (window, document) {
  "use strict";

  var API_BASE = "https://hke-backend.onrender.com";
  var PAGE = (window.location.pathname || "").split("/").pop() || "index.html";
  var STYLE_ID = "hkeDynamicImageStyles";

  function safeJsonParse(text) {
    try {
      return JSON.parse(text);
    } catch (err) {
      return null;
    }
  }

  async function fetchItems(path) {
    try {
      var res = await window.fetch(API_BASE + path);
      var text = await res.text();
      var data = safeJsonParse(text) || {};
      if (!res.ok || data.ok === false) return null;
      return data.items != null ? data.items : data.data;
    } catch (err) {
      console.warn("HKE dynamic content fetch failed for", path, err);
      return null;
    }
  }

  function ensureStyleEl() {
    var styleEl = document.getElementById(STYLE_ID);
    if (styleEl) return styleEl;
    styleEl = document.createElement("style");
    styleEl.id = STYLE_ID;
    document.head.appendChild(styleEl);
    return styleEl;
  }

  function setHeroImage(url) {
    if (!url) return;
    var heroSelector = PAGE === "index.html" ? "header.hero" : ".hero";
    var styleEl = ensureStyleEl();
    styleEl.textContent += "\n" + heroSelector + "::before{background:linear-gradient(90deg, rgba(0,0,0,.82), rgba(0,0,0,.32)),url(\"" + String(url).replace(/"/g, '\\"') + "\") center/cover no-repeat !important;}";
  }

  function formatPrice(value) {
    var raw = String(value || "").trim();
    if (!raw) return "";
    if (/^₹/.test(raw) || /^â‚¹/.test(raw)) return raw;
    return "₹" + raw;
  }

  function splitRoute(route) {
    return String(route || "")
      .split(/,|•|→|->|\|/)
      .map(function (item) { return item.trim(); })
      .filter(Boolean);
  }

  function setText(el, value) {
    if (!el || !value) return;
    el.textContent = value;
  }

  function setImage(el, src, alt) {
    if (!el || !src) return;
    el.src = src;
    if (alt) el.alt = alt;
  }

  function buildAiDestinationMap(items) {
    var map = {};
    (items || []).forEach(function (item) {
      if (!item || typeof item !== "object") return;
      var state = String(item.state || item.destination || "").trim();
      var place = String(item.destination || "").trim();
      if (!state) return;
      if (!map[state]) map[state] = [];
      if (place && map[state].indexOf(place) === -1) {
        map[state].push(place);
      }
    });
    return map;
  }

  function mergeSelectOptions(selectEl, values, placeholder) {
    if (!selectEl || !values || !values.length) return;
    var existing = {};
    Array.prototype.forEach.call(selectEl.options, function (option) {
      existing[String(option.value || "").trim()] = true;
    });

    if (placeholder && !selectEl.options.length) {
      var emptyOption = document.createElement("option");
      emptyOption.value = "";
      emptyOption.textContent = placeholder;
      selectEl.appendChild(emptyOption);
    }

    values.forEach(function (value) {
      var next = String(value || "").trim();
      if (!next || existing[next]) return;
      var option = document.createElement("option");
      option.value = next;
      option.textContent = next;
      selectEl.appendChild(option);
    });
  }

  function applyHomepagePackages(homepagePackages) {
    if (PAGE !== "index.html" || !homepagePackages || typeof homepagePackages !== "object") return;

    document.querySelectorAll("[data-hke-home-package]").forEach(function (card) {
      var key = card.getAttribute("data-hke-home-package");
      var item = homepagePackages[key];
      if (!item) return;

      setText(card.querySelector(".package-title"), item.name || "");
      setText(card.querySelector(".package-text"), item.desc || "");

      var priceEl = card.querySelector(".package-price");
      if (priceEl && item.price) {
        priceEl.innerHTML = formatPrice(item.price) + ' <small>/ person</small>';
      }
    });
  }

  function applyIndexTourPackages(tourPackages) {
    if (PAGE !== "index.html" || !tourPackages || typeof tourPackages !== "object") return;

    document.querySelectorAll("[data-hke-home-package]").forEach(function (card) {
      var key = card.getAttribute("data-hke-home-package");
      var item = tourPackages[key];
      if (!item) return;

      setImage(card.querySelector(".package-img"), item.image, item.title);
      setText(card.querySelector(".package-route"), item.route || "");
      if (item.title) setText(card.querySelector(".package-title"), item.title);
      if (item.desc) setText(card.querySelector(".package-text"), item.desc);
      var priceEl = card.querySelector(".package-price");
      if (priceEl && item.price) {
        priceEl.innerHTML = formatPrice(item.price) + ' <small>/ person</small>';
      }
    });
  }

  function applyIndexImages(images) {
    if (PAGE !== "index.html" || !images || typeof images !== "object") return;

    if (images.homepage) setHeroImage(images.homepage);
    setImage(document.querySelector('[data-hke-image-section="ai-planner"]'), images["ai-planner"], "AI Trip Planner");
    setImage(document.querySelector('[data-hke-image-section="pilgrimage"]'), images.pilgrimage, "Pilgrimage Tours");
    setImage(document.querySelector('[data-hke-image-section="kashmir-package"]'), images["kashmir-package"], "Kashmir Package");
    setImage(document.querySelector('[data-hke-image-section="manali-package"]'), images["manali-package"], "Manali Package");
    setImage(document.querySelector('[data-hke-image-section="kerala-package"]'), images["kerala-package"], "Kerala Package");
    setImage(document.querySelector('[data-hke-image-section="himachal-package"]'), images["himachal-package"], "Himachal Package");
  }

  function applyTourPagePackage(tourPackages) {
    var pageKey = document.body.getAttribute("data-hke-tour-page");
    if (!pageKey || !tourPackages || typeof tourPackages !== "object") return;

    var item = tourPackages[pageKey];
    if (!item) return;

    if (item.title) {
      document.body.setAttribute("data-hke-package-name", item.title);
      setText(document.querySelector(".hero h1"), item.title);
    }
    if (item.price) {
      document.body.setAttribute("data-hke-package-price", String(item.price).replace(/[^\d.]/g, ""));
      var priceEl = document.querySelector(".price-main");
      if (priceEl) priceEl.textContent = formatPrice(item.price);

      var advanceEl = document.querySelector(".advance-box");
      var numericPrice = Number(String(item.price).replace(/[^\d.]/g, ""));
      if (advanceEl && numericPrice) {
        advanceEl.textContent = "20% advance booking: ₹" + Math.round(numericPrice * 0.2).toLocaleString("en-IN") + " per person";
      }
    }
    if (item.desc) {
      setText(document.querySelector(".hero p"), item.desc);
    }
    if (item.route) {
      var routeContainer = document.querySelector(".glass-card .mb-3");
      var chips = routeContainer ? routeContainer.querySelectorAll(".info-chip") : [];
      var routeParts = splitRoute(item.route);
      if (routeContainer && routeParts.length) {
        routeContainer.innerHTML = routeParts.map(function (part) {
          return '<span class="info-chip">' + part + "</span>";
        }).join("");
      } else if (routeContainer && !chips.length) {
        routeContainer.textContent = item.route;
      }
    }
    if (item.image) {
      setHeroImage(item.image);
    }
  }

  function applyAiDestinations(items) {
    var map = buildAiDestinationMap(items || []);
    window.HKE_DYNAMIC_AI_DESTINATION_MAP = map;
    window.dispatchEvent(new CustomEvent("hke:dynamic-ai-destinations", { detail: map }));

    if (PAGE !== "ai-planner.html") return;
    var selectEl = document.getElementById("destination");
    mergeSelectOptions(selectEl, Object.keys(map), "— Select —");
  }

  function renderPilgrimagePackages(items) {
    if (PAGE !== "pilgrimage.html") return;
    var container = document.getElementById("hkeDynamicPilgrimagePackages");
    if (!container || !Array.isArray(items) || !items.length) return;

    container.innerHTML = items.map(function (item) {
      return [
        '<div class="col-md-6 col-xl-4">',
        '  <div class="glass-card h-100">',
        '    <div class="d-inline-block px-3 py-2 rounded-pill mb-3" style="background:rgba(217,178,95,.14);border:1px solid rgba(217,178,95,.18);color:#F0D08A;font-weight:700;">' + String(item.religion || "Pilgrimage") + "</div>",
        '    <h4 style="font-weight:800;color:#f6e4b3;">' + String(item.name || "Pilgrimage Package") + "</h4>",
        '    <div style="font-weight:700;color:#F0D08A;margin:10px 0;">' + formatPrice(item.price || "") + "</div>",
        '    <p style="color:rgba(234,239,247,.76);line-height:1.8;margin:0;">' + String(item.desc || "") + "</p>",
        "  </div>",
        "</div>"
      ].join("");
    }).join("");
  }

  function applyDynamicImages(images) {
    if (!images || typeof images !== "object") return;
    applyIndexImages(images);

    var pageImageSection = document.body.getAttribute("data-hke-image-section");
    if (pageImageSection && images[pageImageSection]) {
      setHeroImage(images[pageImageSection]);
    }
  }

  function onReady(callback) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", callback, { once: true });
      return;
    }
    callback();
  }

  Promise.all([
    fetchItems("/api/public/homepage-packages"),
    fetchItems("/api/public/images"),
    fetchItems("/api/public/ai-destinations"),
    fetchItems("/api/public/tour-packages"),
    fetchItems("/api/public/pilgrimage-packages")
  ]).then(function (results) {
    onReady(function () {
      var homepagePackages = results[0];
      var images = results[1];
      var aiDestinations = results[2];
      var tourPackages = results[3];
      var pilgrimagePackages = results[4];

      applyHomepagePackages(homepagePackages);
      applyIndexTourPackages(tourPackages);
      applyDynamicImages(images);
      applyTourPagePackage(tourPackages);
      applyAiDestinations(aiDestinations);
      renderPilgrimagePackages(pilgrimagePackages);
    });
  }).catch(function (err) {
    console.warn("HKE dynamic content bootstrap failed.", err);
  });
})(window, document);
