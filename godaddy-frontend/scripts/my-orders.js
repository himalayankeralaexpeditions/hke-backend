(function () {
  "use strict";

  var API_BASE = "https://hke-backend.onrender.com";

  function $(id) {
    return document.getElementById(id);
  }

  function formatINR(value) {
    return "Rs " + Math.round(Number(value) || 0).toLocaleString("en-IN");
  }

  function escapeHTML(value) {
    return String(value || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function normalizePhone(value) {
    return String(value || "").replace(/\D/g, "").slice(-10);
  }

  function getProfilePhone() {
    if (window.HKEAuthGate && window.HKEAuthGate.getStoredCustomerProfile) {
      var profile = window.HKEAuthGate.getStoredCustomerProfile() || {};
      if (profile.phone) return normalizePhone(profile.phone);
    }
    return normalizePhone(
      window.localStorage.getItem("HKE_CUSTOMER_PHONE") ||
      window.localStorage.getItem("HKE_OTP_VERIFIED_PHONE") ||
      window.localStorage.getItem("HKE_ORDER_LOOKUP_VALUE") ||
      ""
    );
  }

  function setStatus(message) {
    $("ordersStatus").textContent = message;
  }

  function getFallbackImage(destination) {
    var dest = String(destination || "").toLowerCase();
    if (dest.indexOf("kashmir") !== -1) return "media/kashmir-package.png";
    if (dest.indexOf("manali") !== -1) return "media/manali.jpg";
    if (dest.indexOf("kerala") !== -1) return "media/kerala-package.png";
    if (dest.indexOf("leh") !== -1 || dest.indexOf("ladakh") !== -1) return "media/leh-ladakh2.png";
    return "media/hero-banner.jpg";
  }

  function renderStats(items) {
    var total = items.length;
    var confirmed = items.filter(function (item) {
      return String(item.bookingStatus || "").toLowerCase() === "confirmed";
    }).length;
    var paid = items.filter(function (item) {
      return String(item.paymentStatus || "").toLowerCase() === "paid";
    }).length;
    var due = items.reduce(function (sum, item) {
      return sum + (Number(item.remainingAmount) || 0);
    }, 0);

    $("statTotal").textContent = String(total);
    $("statConfirmed").textContent = String(confirmed);
    $("statPaid").textContent = String(paid);
    $("statDue").textContent = formatINR(due);
  }

  function buildRouteSummary(routeMap, item) {
    var parts = [
      routeMap.startPoint,
      routeMap.destination || item.destination,
      routeMap.endPoint
    ].filter(Boolean);
    return parts.join(" -> ") || item.destination || "-";
  }

  function buildPdfHtml(item) {
    var itinerary = item.latestItinerary || {};
    var days = Array.isArray(itinerary.days) ? itinerary.days : [];
    var routeMap = item.routeMap || {};
    var images = Array.isArray(item.itineraryImages) ? item.itineraryImages : [];
    var coverImage = (images[0] && images[0].imageUrl) || getFallbackImage(item.destination);
    var dayImages = images.slice(1);
    var inclusions = Array.isArray(itinerary.inclusions) ? itinerary.inclusions : [];
    var exclusions = Array.isArray(itinerary.exclusions) ? itinerary.exclusions : [];
    var terms = Array.isArray(itinerary.terms) ? itinerary.terms : [];
    var hotelInfo = item.hotelInfo || {};
    var cabInfo = item.cabInfo || {};

    return [
      '<div style="background:#ffffff;color:#1a1a1a;font-family:Inter,Arial,sans-serif;width:794px;margin:0 auto;">',
      '<section style="padding:48px 52px 32px;min-height:1020px;page-break-after:always;">',
      '<div style="display:flex;align-items:center;gap:16px;margin-bottom:24px;">',
      '<img src="media/logo.png" alt="HKE" style="width:68px;height:68px;object-fit:contain;">',
      '<div><div style="font-size:28px;font-weight:800;color:#d9b25f;">Himalayan Kerala Expeditions</div><div style="font-size:15px;color:#444;">We Plan. We Care.</div></div>',
      '</div>',
      '<img src="' + escapeHTML(coverImage) + '" alt="Cover" style="width:100%;height:340px;object-fit:cover;border-radius:18px;border:1px solid #e9d6aa;">',
      '<h1 style="font-size:32px;line-height:1.1;margin:28px 0 12px;color:#c8952a;">' + escapeHTML(item.packageName || item.destination || "HKE Itinerary") + '</h1>',
      '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px 24px;margin-top:20px;font-size:14px;line-height:1.7;">',
      '<div><strong>Customer Name:</strong> ' + escapeHTML(item.customerName || "-") + '</div>',
      '<div><strong>Travel Dates:</strong> ' + escapeHTML((item.startDate || "-") + " to " + (item.endDate || "-")) + '</div>',
      '<div><strong>Travellers:</strong> ' + escapeHTML(String(item.travellers || "-")) + '</div>',
      '<div><strong>Rooms:</strong> ' + escapeHTML(String(item.rooms || "-")) + '</div>',
      '<div style="grid-column:1/-1;"><strong>Route Summary:</strong> ' + escapeHTML(buildRouteSummary(routeMap, item)) + '</div>',
      '</div>',
      '<div style="margin-top:26px;padding:18px 20px;background:#fbf8f1;border:1px solid #ead8af;border-radius:16px;font-size:14px;line-height:1.8;">' + escapeHTML(itinerary.summary || "Your itinerary has been prepared by HKE.") + '</div>',
      '</section>',
      days.map(function (day, index) {
        var imageUrl = (dayImages[index] && dayImages[index].imageUrl) || coverImage;
        var activities = Array.isArray(day.activities) && day.activities.length ? day.activities : ["No activities listed."];
        return [
          '<section style="padding:44px 52px 32px;min-height:1020px;page-break-after:always;">',
          '<div style="font-size:24px;font-weight:800;color:#c8952a;margin-bottom:10px;">Day ' + escapeHTML(day.day) + ' - ' + escapeHTML(day.title || "") + '</div>',
          '<div style="font-size:14px;color:#444;margin-bottom:16px;line-height:1.7;"><strong>Route:</strong> ' + escapeHTML(day.route || "-") + '</div>',
          '<img src="' + escapeHTML(imageUrl) + '" alt="Day image" style="width:100%;height:260px;object-fit:cover;border-radius:16px;border:1px solid #ead8af;margin-bottom:18px;">',
          '<div style="font-size:14px;line-height:1.8;color:#1f1f1f;"><strong>Activities</strong></div>',
          '<ul style="margin:8px 0 16px 18px;padding:0;font-size:14px;line-height:1.9;color:#333;">' + activities.map(function (activity) {
            return '<li>' + escapeHTML(activity) + '</li>';
          }).join("") + '</ul>',
          '<div style="padding:14px 16px;background:#fbf8f1;border:1px solid #ead8af;border-radius:14px;font-size:13px;line-height:1.8;color:#444;"><strong>Notes:</strong> ' + escapeHTML(day.notes || "Sightseeing flow may adjust slightly based on local conditions and guest comfort.") + '</div>',
          '</section>'
        ].join("");
      }).join(""),
      '<section style="padding:44px 52px 40px;min-height:900px;">',
      '<div style="font-size:24px;font-weight:800;color:#c8952a;margin-bottom:16px;">Trip Support Details</div>',
      '<div style="margin-bottom:18px;font-size:14px;line-height:1.8;"><strong>Google Maps Route:</strong> ' + escapeHTML(routeMap.googleMapsSearchUrl || "-") + '<br><strong>Open Directions:</strong> ' + escapeHTML(routeMap.googleMapsDirectionsUrl || "-") + '</div>',
      '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px;">',
      '<div style="padding:16px;border:1px solid #ead8af;border-radius:16px;background:#fbf8f1;"><div style="font-weight:800;color:#c8952a;margin-bottom:8px;">Hotel Details</div><div style="font-size:13px;line-height:1.8;"><strong>Name:</strong> ' + escapeHTML(hotelInfo.name || "To be assigned by HKE") + '<br><strong>Location:</strong> ' + escapeHTML(hotelInfo.location || item.destination || "-") + '<br><strong>Check-in:</strong> ' + escapeHTML(hotelInfo.checkInDate || item.startDate || "-") + '<br><strong>Check-out:</strong> ' + escapeHTML(hotelInfo.checkOutDate || item.endDate || "-") + '</div></div>',
      '<div style="padding:16px;border:1px solid #ead8af;border-radius:16px;background:#fbf8f1;"><div style="font-weight:800;color:#c8952a;margin-bottom:8px;">Cab Details</div><div style="font-size:13px;line-height:1.8;"><strong>Driver:</strong> ' + escapeHTML(cabInfo.driverName || "To be assigned by HKE") + '<br><strong>Vehicle:</strong> ' + escapeHTML(cabInfo.vehicle || "Vehicle to be assigned by HKE") + '<br><strong>Pickup:</strong> ' + escapeHTML(cabInfo.pickupLocation || "-") + '<br><strong>Pickup Date:</strong> ' + escapeHTML(cabInfo.pickupDate || item.startDate || "-") + '</div></div>',
      '</div>',
      '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;">',
      '<div><div style="font-size:18px;font-weight:800;color:#c8952a;margin-bottom:8px;">Inclusions</div><ul style="margin:0;padding-left:18px;font-size:13px;line-height:1.9;">' + (inclusions.length ? inclusions.map(function (line) { return '<li>' + escapeHTML(line) + '</li>'; }).join("") : '<li>As per booking confirmation</li>') + '</ul></div>',
      '<div><div style="font-size:18px;font-weight:800;color:#c8952a;margin-bottom:8px;">Exclusions</div><ul style="margin:0;padding-left:18px;font-size:13px;line-height:1.9;">' + (exclusions.length ? exclusions.map(function (line) { return '<li>' + escapeHTML(line) + '</li>'; }).join("") : '<li>Personal expenses and items not listed</li>') + '</ul></div>',
      '<div><div style="font-size:18px;font-weight:800;color:#c8952a;margin-bottom:8px;">Terms</div><ul style="margin:0;padding-left:18px;font-size:13px;line-height:1.9;">' + (terms.length ? terms.map(function (line) { return '<li>' + escapeHTML(line) + '</li>'; }).join("") : '<li>Travel schedules may vary based on local conditions</li>') + '</ul></div>',
      '</div>',
      '<div style="margin-top:28px;padding-top:18px;border-top:1px solid #ddd;font-size:13px;line-height:1.9;color:#333;"><strong>We Plan. We Care.</strong><br>WhatsApp: +91 9797294747<br>Email: info@himalayankeralaexpeditions.com</div>',
      '</section>',
      '</div>'
    ].join("");
  }

  function printFallback(item) {
    var popup = window.open("", "_blank", "noopener,noreferrer,width=1020,height=760");
    if (!popup) {
      window.alert("Please allow popups to print the itinerary.");
      return;
    }
    popup.document.open();
    popup.document.write("<!DOCTYPE html><html><head><meta charset='utf-8'><title>HKE Itinerary</title><style>body{margin:0;background:#fff} img{max-width:100%} @media print{section{page-break-after:always}}</style></head><body>" + buildPdfHtml(item) + "</body></html>");
    popup.document.close();
    popup.focus();
    window.setTimeout(function () { popup.print(); }, 500);
  }

  function downloadItineraryPdf(item) {
    if (!window.html2pdf) {
      printFallback(item);
      return;
    }
    var itinerary = item.latestItinerary || {};
    var days = Array.isArray(itinerary.days) ? itinerary.days : [];
    if (!days.length) {
      window.alert("Itinerary PDF is available only after the booked itinerary is saved.");
      return;
    }

    var container = document.createElement("div");
    container.style.position = "fixed";
    container.style.left = "-99999px";
    container.style.top = "0";
    container.style.width = "794px";
    container.style.background = "#ffffff";
    container.innerHTML = buildPdfHtml(item);
    document.body.appendChild(container);

    var fileName = (item.bookingRef || "HKE-Itinerary").replace(/[^a-zA-Z0-9_-]/g, "_") + ".pdf";
    window.html2pdf().set({
      margin: 0,
      filename: fileName,
      image: { type: "jpeg", quality: 0.95 },
      html2canvas: { scale: 2, useCORS: true, backgroundColor: "#ffffff" },
      jsPDF: { unit: "pt", format: "a4", orientation: "portrait" },
      pagebreak: { mode: ["css", "legacy"] }
    }).from(container).save().catch(function () {
      printFallback(item);
    }).finally(function () {
      if (container.parentNode) container.parentNode.removeChild(container);
    });
  }

  function renderDayList(itinerary) {
    var days = Array.isArray((itinerary || {}).days) ? itinerary.days : [];
    if (!days.length) {
      return '<div class="sub-card"><h3>Itinerary</h3><div class="value">Your itinerary is being prepared by HKE.</div></div>';
    }
    return [
      '<div class="sub-card"><h3>Saved Itinerary</h3>',
      '<div class="sub-list">' + days.map(function (day) {
        var activities = Array.isArray(day.activities) ? day.activities.join(", ") : "";
        return "<p><strong>Day " + escapeHTML(day.day) + ":</strong> " + escapeHTML(day.title || "") + "<br>" +
          escapeHTML(day.route || "") + (activities ? "<br>" + escapeHTML(activities) : "") + "</p>";
      }).join("") + "</div></div>"
    ].join("");
  }

  function renderMapBlock(routeMap) {
    if (!routeMap || !(routeMap.destination || routeMap.endPoint || routeMap.origin || routeMap.startPoint)) {
      return "";
    }
    var bookingRef = escapeHTML(routeMap.bookingRef || routeMap._bookingRef || "");
    return [
      '<div class="sub-card">',
      "<h3>Route Map</h3>",
      '<div class="route-map-shell">',
      '<div class="booking-actions">',
      '<a class="btn-ghost js-route-full" id="route-full-' + bookingRef + '" href="' + escapeHTML(routeMap.googleMapsSearchUrl) + '" target="_blank" rel="noopener">View Full Route</a>',
      '<a class="btn-ghost js-route-complete" id="route-complete-' + bookingRef + '" href="' + escapeHTML(routeMap.googleMapsDirectionsUrl || routeMap.googleMapsSearchUrl) + '" target="_blank" rel="noopener">Open Complete Route in Google Maps</a>',
      "</div>",
      '<div class="route-map-toolbar">',
      '<select id="route-day-select-' + bookingRef + '" aria-label="Select day route"></select>',
      '<a class="btn-ghost js-route-day" id="route-day-' + bookingRef + '" href="#" target="_blank" rel="noopener">Open Day Route</a>',
      "</div>",
      '<div class="route-map-message" id="route-message-' + bookingRef + '">Interactive Google Maps is unavailable. Showing fallback route preview.</div>',
      '<div class="route-map-canvas" id="route-map-' + bookingRef + '"></div>',
      '<div class="route-map-legend" id="route-legend-' + bookingRef + '"></div>',
      '<iframe class="map-frame" id="route-frame-' + bookingRef + '" loading="lazy" referrerpolicy="no-referrer-when-downgrade" src="https://www.google.com/maps?q=' + encodeURIComponent(routeMap.destination || routeMap.endPoint || "") + '&output=embed"></iframe>',
      "</div>",
      "</div>"
    ].join("");
  }

  function enhanceRouteMaps(items) {
    if (!window.HKERouteMap || !Array.isArray(items)) return;

    items.forEach(function (item) {
      var bookingRef = String(item.bookingRef || "").trim();
      if (!bookingRef) return;

      var routeMap = Object.assign({}, item.routeMap || {}, {
        bookingRef: bookingRef,
        origin: (item.routeMap && item.routeMap.origin) || (item.routeMap && item.routeMap.startPoint) || item.fromLocation || "",
        startPoint: (item.routeMap && item.routeMap.startPoint) || item.fromLocation || "",
        destination: (item.routeMap && item.routeMap.destination) || item.destination || "",
        endPoint: (item.routeMap && item.routeMap.endPoint) || item.endPoint || item.destination || ""
      });

      window.HKERouteMap.render({
        routeMap: routeMap,
        itinerary: item.latestItinerary || {},
        customer: {
          fromLocation: item.fromLocation || "",
          destination: item.destination || "",
          endPoint: item.endPoint || item.destination || "",
          places: routeMap.places || []
        },
        summaryElement: null,
        mapElement: $("route-map-" + bookingRef),
        legendElement: $("route-legend-" + bookingRef),
        daySelectElement: $("route-day-select-" + bookingRef),
        viewFullRouteButton: $("route-full-" + bookingRef),
        openDayRouteButton: $("route-day-" + bookingRef),
        openCompleteRouteButton: $("route-complete-" + bookingRef),
        fallbackFrame: $("route-frame-" + bookingRef),
        messageElement: $("route-message-" + bookingRef)
      });
    });
  }

  function renderSupportBlock(item) {
    var hotel = item.hotelInfo || {};
    var cab = item.cabInfo || {};
    return [
      '<div class="sub-card"><h3>Hotel Details</h3>',
      '<div class="detail-grid">',
      '<div class="detail"><div class="label">Hotel</div><div class="value">' + escapeHTML(hotel.name || "To be assigned by HKE") + "</div></div>",
      '<div class="detail"><div class="label">Location</div><div class="value">' + escapeHTML(hotel.location || item.destination || "-") + "</div></div>",
      '<div class="detail"><div class="label">Check-in</div><div class="value">' + escapeHTML(hotel.checkInDate || item.startDate || "-") + "</div></div>",
      '<div class="detail"><div class="label">Check-out</div><div class="value">' + escapeHTML(hotel.checkOutDate || item.endDate || "-") + "</div></div>",
      "</div>",
      '<div class="booking-actions">' + (hotel.googleMapsUrl ? '<a class="btn-ghost" href="' + escapeHTML(hotel.googleMapsUrl) + '" target="_blank" rel="noopener">Hotel Location</a>' : "") + "</div>",
      "</div>",
      '<div class="sub-card"><h3>Cab Details</h3>',
      '<div class="detail-grid">',
      '<div class="detail"><div class="label">Driver</div><div class="value">' + escapeHTML(cab.driverName || "To be assigned by HKE") + "</div></div>",
      '<div class="detail"><div class="label">Vehicle</div><div class="value">' + escapeHTML(cab.vehicle || "Vehicle to be assigned by HKE") + "</div></div>",
      '<div class="detail"><div class="label">Pickup</div><div class="value">' + escapeHTML(cab.pickupLocation || "-") + "</div></div>",
      '<div class="detail"><div class="label">Pickup Date</div><div class="value">' + escapeHTML(cab.pickupDate || item.startDate || "-") + "</div></div>",
      "</div>",
      '<div class="booking-actions">' + (cab.pickupLocation ? '<a class="btn-ghost" href="https://www.google.com/maps/search/?api=1&query=' + encodeURIComponent(cab.pickupLocation) + '" target="_blank" rel="noopener">Cab Pickup</a>' : "") + "</div>",
      "</div>"
    ].join("");
  }

  function renderItems(items) {
    var list = $("ordersList");
    if (!items.length) {
      $("ordersEmpty").style.display = "block";
      list.innerHTML = "";
      return;
    }

    $("ordersEmpty").style.display = "none";
    list.innerHTML = items.map(function (item) {
      var isFullyPaid = Number(item.remainingAmount) <= 0;
      var routeMap = Object.assign({}, item.routeMap || {}, {
        _bookingRef: item.bookingRef || "",
        origin: (item.routeMap && item.routeMap.origin) || (item.routeMap && item.routeMap.startPoint) || item.fromLocation || "",
        startPoint: (item.routeMap && item.routeMap.startPoint) || item.fromLocation || "",
        destination: (item.routeMap && item.routeMap.destination) || item.destination || "",
        endPoint: (item.routeMap && item.routeMap.endPoint) || item.endPoint || item.destination || ""
      });
      return [
        '<article class="booking-card">',
        '<h2>' + escapeHTML(item.packageName || item.destination || "HKE Booking") + "</h2>",
        '<div class="badges">',
        '<span class="badge amber">' + escapeHTML(item.bookingStatus || "pending") + "</span>",
        '<span class="badge green">' + escapeHTML(item.paymentStatus || "payment_pending") + "</span>",
        '<span class="badge amber">' + escapeHTML(item.itineraryStatus || "pending") + " itinerary</span>",
        "</div>",
        '<div class="detail-grid">',
        '<div class="detail"><div class="label">Booking Ref</div><div class="value">' + escapeHTML(item.bookingRef) + "</div></div>",
        '<div class="detail"><div class="label">Destination</div><div class="value">' + escapeHTML(item.destination || "-") + "</div></div>",
        '<div class="detail"><div class="label">Travel Dates</div><div class="value">' + escapeHTML((item.startDate || "-") + " to " + (item.endDate || "-")) + "</div></div>",
        '<div class="detail"><div class="label">Travellers / Rooms</div><div class="value">' + escapeHTML(String(item.travellers || 0) + " / " + String(item.rooms || 0)) + "</div></div>",
        '<div class="detail"><div class="label">Total Amount</div><div class="value">' + escapeHTML(formatINR(item.totalAmount)) + "</div></div>",
        '<div class="detail"><div class="label">Paid Amount</div><div class="value">' + escapeHTML(formatINR(item.paidAmount)) + "</div></div>",
        '<div class="detail"><div class="label">Remaining Amount</div><div class="value">' + escapeHTML(formatINR(item.remainingAmount)) + "</div></div>",
        '<div class="detail"><div class="label">Balance Due</div><div class="value">' + escapeHTML(item.fullPaymentDeadline || "7 days before travel") + "</div></div>",
        "</div>",
        '<div class="booking-actions">',
        '<button class="placeholder-btn js-view-itinerary" type="button" data-booking-ref="' + escapeHTML(item.bookingRef || "") + '">View Itinerary</button>',
        '<button class="placeholder-btn js-download" type="button" data-booking-ref="' + escapeHTML(item.bookingRef || "") + '">Download Premium PDF</button>',
        '<button class="placeholder-btn js-remaining" type="button" data-remaining="' + escapeHTML(item.remainingAmount) + '">' + (isFullyPaid ? "Fully Paid" : "Pay Remaining") + "</button>",
        "</div>",
        renderMapBlock(routeMap),
        renderSupportBlock(item),
        '<div id="itinerary-' + escapeHTML(item.bookingRef || "") + '"></div>',
        renderDayList(item.latestItinerary || {}),
        "</article>"
      ].join("");
    }).join("");

    Array.prototype.forEach.call(list.querySelectorAll(".js-view-itinerary"), function (button) {
      button.addEventListener("click", function () {
        var target = document.getElementById("itinerary-" + button.getAttribute("data-booking-ref"));
        if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    });
    Array.prototype.forEach.call(list.querySelectorAll(".js-download"), function (button, index) {
      button.addEventListener("click", function () {
        downloadItineraryPdf(items[index]);
      });
    });
    Array.prototype.forEach.call(list.querySelectorAll(".js-remaining"), function (button) {
      button.addEventListener("click", function () {
        var remaining = Number(button.getAttribute("data-remaining") || 0);
        if (remaining <= 0) {
          window.alert("This booking is already fully paid.");
          return;
        }
        window.alert("Online remaining-balance payment will be enabled here soon. Please contact HKE support for immediate settlement.");
      });
    });

    enhanceRouteMaps(items);
  }

  async function init() {
    var phone = getProfilePhone();
    if (!phone) {
      setStatus("No verified phone number was found. Please log in with OTP to view your bookings.");
      if (window.HKEAuthGate && window.HKEAuthGate.requireLogin) {
        window.HKEAuthGate.requireLogin("my-orders.html");
      }
      return;
    }

    try {
      setStatus("Loading bookings for " + phone + "...");
      var response = await window.fetch(API_BASE + "/api/customer/bookings?phone=" + encodeURIComponent(phone));
      var data = {};
      try {
        data = await response.json();
      } catch (_err) {
        data = {};
      }
      if (!response.ok || data.ok === false) {
        throw new Error(data.detail || data.message || "Unable to load bookings.");
      }
      var items = Array.isArray(data.items) ? data.items : [];
      renderStats(items);
      renderItems(items);
      setStatus(items.length ? "Latest MongoDB bookings, itinerary, route map, hotel details, and cab details loaded successfully." : "No bookings found for your verified phone number yet.");
    } catch (error) {
      setStatus(error.message || "Unable to load bookings right now.");
      $("ordersList").innerHTML = "";
      $("ordersEmpty").style.display = "block";
      renderStats([]);
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
