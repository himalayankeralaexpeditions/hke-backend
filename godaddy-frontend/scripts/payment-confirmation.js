(function () {
  "use strict";

  var CONFIRMATION_KEY = "hkeLastConfirmedBooking";
  var PAYMENT_SUCCESS_KEY = "hkeLastPaymentSuccess";
  var BOOKING_CONTEXT_KEY = "HKE_BOOKING_CONTEXT";

  function $(id) {
    return document.getElementById(id);
  }

  function formatINR(value) {
    return "Rs " + Math.round(Number(value) || 0).toLocaleString("en-IN");
  }

  function safeJSONParse(raw) {
    if (!raw) return null;
    try {
      return JSON.parse(raw);
    } catch (_err) {
      return null;
    }
  }

  function readData() {
    return (
      safeJSONParse(window.localStorage.getItem(CONFIRMATION_KEY)) ||
      safeJSONParse(window.localStorage.getItem(PAYMENT_SUCCESS_KEY)) ||
      safeJSONParse(window.localStorage.getItem(BOOKING_CONTEXT_KEY))
    );
  }

  function setText(id, value) {
    $(id).textContent = value || "-";
  }

  function init() {
    var data = readData();
    if (!data) {
      $("confirmationEmpty").style.display = "block";
      return;
    }

    $("confirmationShell").style.display = "block";
    setText("bookingRefValue", data.bookingRef);
    setText("paymentIdValue", data.razorpayPaymentId);
    setText("paidAmountValue", formatINR(data.paidAmount));
    setText("remainingAmountValue", formatINR(data.remainingAmount));
    setText("paymentStatusValue", data.paymentStatus || "confirmed");
    setText("packageValue", data.packageName);
    setText("destinationValue", data.destination);
    setText("datesValue", (data.startDate || "-") + " to " + (data.endDate || "-"));
    setText("travellersValue", String(data.travellers || "-"));
    setText("roomsValue", String(data.rooms || "-"));
    setText("customerValue", [data.customerName, data.customerPhone].filter(Boolean).join(" | "));
    if (Number(data.remainingAmount) > 0) {
      setText("paymentStatusValue", "Partially paid - itinerary saved in My Bookings");
    }
    if (data.sheetWarning) {
      $("sheetWarningBox").textContent = data.sheetWarning;
      $("sheetWarningBox").style.display = "block";
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
