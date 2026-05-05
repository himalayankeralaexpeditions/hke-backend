(function () {
  "use strict";

  console.log("HKE route map script loaded");
  console.log("HKE config found", Boolean(window.HKE_CONFIG));
  console.log("Google Maps key present", Boolean(window.HKE_CONFIG && window.HKE_CONFIG.GOOGLE_MAPS_API_KEY));

  var DAY_COLORS = ["#f59e0b", "#14b8a6", "#60a5fa", "#a78bfa", "#fb7185"];
  var mapsLoaderPromise = null;
  var mapsLoadFailure = null;

  function uniqueList(values) {
    var seen = Object.create(null);
    return (Array.isArray(values) ? values : []).map(function (value) {
      return String(value || "").trim();
    }).filter(function (value) {
      if (!value) return false;
      var key = value.toLowerCase();
      if (seen[key]) return false;
      seen[key] = true;
      return true;
    });
  }

  function safeRouteString(value) {
    return String(value || "").trim();
  }

  function normalizeDayColor(index, explicitColor) {
    return explicitColor || DAY_COLORS[index % DAY_COLORS.length];
  }

  function createMapsError(code, message, originalError) {
    var error = originalError instanceof Error ? originalError : new Error(message);
    error.code = code;
    error.userMessage = message;
    return error;
  }

  function classifyMapsError(error) {
    var source = String(
      (error && (error.code || error.userMessage || error.message)) ||
      ""
    ).toLowerCase();

    if (source.indexOf("missing") !== -1 && source.indexOf("key") !== -1) {
      return createMapsError("MISSING_API_KEY", "Missing API key", error);
    }
    if (source.indexOf("referer") !== -1 || source.indexOf("referrer") !== -1) {
      return createMapsError("REFERER_NOT_ALLOWED", "Referer not allowed", error);
    }
    if (source.indexOf("billing") !== -1) {
      return createMapsError("BILLING_ISSUE", "Billing issue", error);
    }
    if (source.indexOf("activated") !== -1 || source.indexOf("api not activated") !== -1) {
      return createMapsError("API_NOT_ACTIVATED", "API not activated", error);
    }
    if (source.indexOf("request_denied") !== -1 || source.indexOf("denied") !== -1) {
      return createMapsError("REQUEST_DENIED", "Referer not allowed / API not activated / Billing issue", error);
    }

    return createMapsError("GOOGLE_MAPS_LOAD_FAILED", error && (error.userMessage || error.message) || "Google Maps failed to load", error);
  }

  function getVisibleErrorText(error) {
    var classified = classifyMapsError(error);
    return "Interactive Google Maps is unavailable. " + classified.userMessage + ". Showing fallback route preview.";
  }

  function buildGoogleMapsDirectionsUrl(origin, destination, waypoints) {
    var params = [
      "api=1",
      "travelmode=driving",
      "origin=" + encodeURIComponent(origin || ""),
      "destination=" + encodeURIComponent(destination || "")
    ];

    var cleanWaypoints = uniqueList(waypoints);
    if (cleanWaypoints.length) {
      params.push("waypoints=" + encodeURIComponent(cleanWaypoints.join("|")));
    }

    return "https://www.google.com/maps/dir/?" + params.join("&");
  }

  function buildGoogleMapsSearchUrl(routeMap) {
    var places = uniqueList([routeMap.origin, routeMap.destination, routeMap.endPoint].concat(routeMap.places || []));
    return "https://www.google.com/maps/search/?api=1&query=" + encodeURIComponent(places.join(", ") || routeMap.destination || routeMap.endPoint || "");
  }

  function splitPlacesAcrossDays(places, dayCount) {
    var cleanPlaces = uniqueList(places);
    var count = Math.max(1, Number(dayCount) || 1);
    var chunks = [];
    var startIndex = 0;

    for (var i = 0; i < count; i += 1) {
      var remainingPlaces = cleanPlaces.length - startIndex;
      var remainingDays = count - i;
      var size = remainingPlaces > 0 ? Math.ceil(remainingPlaces / remainingDays) : 0;
      chunks.push(cleanPlaces.slice(startIndex, startIndex + size));
      startIndex += size;
    }

    return chunks;
  }

  function normalizeRouteMap(routeMap, options) {
    var input = routeMap && typeof routeMap === "object" ? routeMap : {};
    var context = options && typeof options === "object" ? options : {};
    var customer = context.customer || {};
    var itinerary = context.itinerary || {};
    var itineraryDays = Array.isArray(itinerary.days) ? itinerary.days : [];
    var places = uniqueList(
      input.places ||
      customer.places ||
      context.places ||
      []
    );
    var origin = safeRouteString(input.origin || input.startPoint || customer.fromLocation || context.origin || "");
    var destination = safeRouteString(input.destination || customer.destination || itinerary.meta && itinerary.meta.destination || context.destination || places[places.length - 1] || origin);
    var endPoint = safeRouteString(input.endPoint || customer.endPoint || context.endPoint || destination);
    var explicitDayRoutes = Array.isArray(input.dayRoutes) ? input.dayRoutes : [];
    var dayRoutes = [];

    if (explicitDayRoutes.length) {
      dayRoutes = explicitDayRoutes.map(function (route, index) {
        var nextOrigin = safeRouteString(route.origin || (index === 0 ? origin : explicitDayRoutes[index - 1].destination) || origin);
        var nextDestination = safeRouteString(route.destination || endPoint || destination || nextOrigin);
        var nextWaypoints = uniqueList(route.waypoints || []);
        return {
          day: Number(route.day) || index + 1,
          title: safeRouteString(route.title || (itineraryDays[index] && itineraryDays[index].title) || ("Day " + (index + 1))),
          origin: nextOrigin,
          destination: nextDestination,
          waypoints: nextWaypoints,
          color: normalizeDayColor(index, route.color),
          googleMapsDirectionsUrl: buildGoogleMapsDirectionsUrl(nextOrigin, nextDestination, nextWaypoints)
        };
      });
    } else {
      var fallbackDayCount = itineraryDays.length || Math.max(1, Math.min(places.length || 1, 5));
      var chunks = splitPlacesAcrossDays(places, fallbackDayCount);
      var previousStop = origin || destination || endPoint;

      dayRoutes = chunks.map(function (chunk, index) {
        var isLastDay = index === chunks.length - 1;
        var cleanChunk = uniqueList(chunk);
        var destinationStop;
        var waypoints;

        if (isLastDay && endPoint) {
          destinationStop = endPoint;
          waypoints = cleanChunk;
        } else if (cleanChunk.length) {
          destinationStop = cleanChunk[cleanChunk.length - 1];
          waypoints = cleanChunk.slice(0, -1);
        } else {
          destinationStop = isLastDay ? (endPoint || destination || previousStop) : (destination || previousStop);
          waypoints = [];
        }

        var route = {
          day: index + 1,
          title: safeRouteString((itineraryDays[index] && itineraryDays[index].title) || ("Day " + (index + 1))),
          origin: previousStop || origin || destinationStop,
          destination: destinationStop || previousStop || endPoint || destination,
          waypoints: waypoints,
          color: normalizeDayColor(index),
          googleMapsDirectionsUrl: ""
        };

        route.googleMapsDirectionsUrl = buildGoogleMapsDirectionsUrl(route.origin, route.destination, route.waypoints);
        previousStop = route.destination;
        return route;
      });
    }

    if (!dayRoutes.length) {
      dayRoutes = [{
        day: 1,
        title: "Day 1",
        origin: origin || destination,
        destination: endPoint || destination || origin,
        waypoints: [],
        color: DAY_COLORS[0],
        googleMapsDirectionsUrl: buildGoogleMapsDirectionsUrl(origin || destination, endPoint || destination || origin, [])
      }];
    }

    return {
      origin: origin,
      destination: destination,
      endPoint: endPoint,
      places: places,
      dayRoutes: dayRoutes,
      googleMapsSearchUrl: input.googleMapsSearchUrl || buildGoogleMapsSearchUrl({
        origin: origin,
        destination: destination,
        endPoint: endPoint,
        places: places
      }),
      googleMapsDirectionsUrl: input.googleMapsDirectionsUrl || buildGoogleMapsDirectionsUrl(origin || destination, endPoint || destination || origin, places)
    };
  }

  function ensureMapsLoaded() {
    var apiKey = window.HKE_CONFIG && window.HKE_CONFIG.GOOGLE_MAPS_API_KEY;

    if (window.google && window.google.maps && typeof window.google.maps.DirectionsService === "function") {
      console.log("Google Maps loaded");
      return Promise.resolve(window.google.maps);
    }

    if (!apiKey) {
      return Promise.reject(createMapsError("MISSING_API_KEY", "Missing API key"));
    }

    if (mapsLoaderPromise) {
      return mapsLoaderPromise;
    }

    mapsLoaderPromise = new Promise(function (resolve, reject) {
      var existingScript = document.querySelector('script[data-hke-google-maps="true"]');
      var settled = false;

      function settleError(error) {
        if (settled) return;
        settled = true;
        mapsLoadFailure = classifyMapsError(error);
        console.log("Google Maps load failed", mapsLoadFailure);
        reject(mapsLoadFailure);
      }

      function settleSuccess() {
        if (settled) return;
        settled = true;
        mapsLoadFailure = null;
        console.log("Google Maps loaded");
        resolve(window.google.maps);
      }

      window.__hkeGoogleMapsInit = function () {
        settleSuccess();
      };

      window.gm_authFailure = function () {
        settleError(createMapsError("REFERER_NOT_ALLOWED", "Referer not allowed"));
      };

      if (existingScript) {
        existingScript.addEventListener("error", function () {
          settleError(createMapsError("GOOGLE_MAPS_LOAD_FAILED", "Google Maps failed to load"));
        }, { once: true });
        return;
      }

      var script = document.createElement("script");
      script.async = true;
      script.defer = true;
      script.dataset.hkeGoogleMaps = "true";
      console.log("Loading Google Maps JS API");
      script.src = "https://maps.googleapis.com/maps/api/js?key=" + encodeURIComponent(apiKey) + "&callback=__hkeGoogleMapsInit";
      script.onerror = function () {
        settleError(createMapsError("GOOGLE_MAPS_LOAD_FAILED", "Google Maps failed to load"));
      };
      document.head.appendChild(script);
    }).catch(function (error) {
      mapsLoaderPromise = null;
      throw error;
    });

    return mapsLoaderPromise;
  }

  function requestDirections(directionsService, route) {
    return new Promise(function (resolve, reject) {
      directionsService.route({
        origin: route.origin,
        destination: route.destination,
        waypoints: uniqueList(route.waypoints).map(function (waypoint) {
          return { location: waypoint, stopover: true };
        }),
        travelMode: window.google.maps.TravelMode.DRIVING
      }, function (result, status) {
        if (status === "OK") {
          resolve(result);
          return;
        }
        if (status === "REQUEST_DENIED") {
          reject(createMapsError("REQUEST_DENIED", "Referer not allowed / API not activated / Billing issue"));
          return;
        }
        reject(new Error("Directions lookup failed for day " + route.day + " (" + status + ")."));
      });
    });
  }

  function buildLegendHtml(routeMap) {
    return routeMap.dayRoutes.map(function (route) {
      var waypointText = route.waypoints.length ? route.waypoints.join(", ") : "Direct drive";
      return [
        '<div class="hke-route-legend-item">',
        '<span class="hke-route-legend-swatch" style="background:' + route.color + ';"></span>',
        '<div class="hke-route-legend-copy">',
        '<strong>Day ' + route.day + "</strong>",
        '<div>' + route.origin + " -> " + route.destination + "</div>",
        '<div class="hke-route-legend-waypoints">Waypoints: ' + waypointText + "</div>",
        "</div>",
        "</div>"
      ].join("");
    }).join("");
  }

  function toggleFallback(options, shouldShow) {
    if (options.mapElement) {
      options.mapElement.style.display = shouldShow ? "none" : "block";
    }
    if (options.legendElement) {
      options.legendElement.style.display = shouldShow ? "none" : "grid";
    }
    if (options.daySelectElement) {
      options.daySelectElement.disabled = shouldShow;
    }
    if (options.fallbackFrame) {
      options.fallbackFrame.style.display = shouldShow ? "block" : "none";
    }
    if (options.messageElement) {
      options.messageElement.style.display = shouldShow ? "block" : "none";
    }
  }

  function updateMessage(options, error) {
    if (!options.messageElement) return;
    options.messageElement.textContent = error ? getVisibleErrorText(error) : "";
  }

  function buildMarkerIcon(color) {
    return {
      path: window.google.maps.SymbolPath.CIRCLE,
      fillColor: color,
      fillOpacity: 1,
      scale: 7,
      strokeColor: "#ffffff",
      strokeWeight: 2
    };
  }

  function addRouteMarkers(map, routeMap) {
    var markers = [];
    var firstDay = routeMap.dayRoutes[0];
    var lastDay = routeMap.dayRoutes[routeMap.dayRoutes.length - 1];
    var geocoder = new window.google.maps.Geocoder();

    function pushMarker(positionName, title, color, label) {
      if (!positionName) return;
      var marker = new window.google.maps.Marker({
        map: map,
        position: { lat: 0, lng: 0 },
        title: title,
        label: label,
        icon: buildMarkerIcon(color)
      });
      markers.push(marker);
      geocoder.geocode({ address: positionName }, function (results, status) {
        if (status === "OK" && results && results[0]) {
          marker.setPosition(results[0].geometry.location);
        }
      });
    }

    pushMarker(firstDay && firstDay.origin, "Start", "#22c55e", "S");

    routeMap.dayRoutes.forEach(function (dayRoute) {
      uniqueList(dayRoute.waypoints).forEach(function (waypoint, index) {
        pushMarker(waypoint, "Day " + dayRoute.day + " Stop " + (index + 1), dayRoute.color, String(dayRoute.day));
      });
    });

    pushMarker(lastDay && lastDay.destination, "End", "#ef4444", "E");

    return markers;
  }

  function renderRouteMap(options) {
    var routeMap = normalizeRouteMap(options.routeMap, {
      customer: options.customer,
      itinerary: options.itinerary,
      origin: options.origin,
      destination: options.destination,
      endPoint: options.endPoint,
      places: options.places
    });
    var daySelect = options.daySelectElement;

    if (options.summaryElement) {
      options.summaryElement.textContent = "Route: " + (routeMap.origin || "-") + " to " + (routeMap.endPoint || routeMap.destination || "-") + ". Stops: " + (routeMap.places.join(", ") || routeMap.destination || "-");
    }

    if (options.legendElement) {
      options.legendElement.innerHTML = buildLegendHtml(routeMap);
    }

    if (options.viewFullRouteButton) {
      options.viewFullRouteButton.href = routeMap.googleMapsSearchUrl || "#";
      options.viewFullRouteButton.textContent = "View Full Route";
    }
    if (options.openCompleteRouteButton) {
      options.openCompleteRouteButton.href = routeMap.googleMapsDirectionsUrl || routeMap.googleMapsSearchUrl || "#";
      options.openCompleteRouteButton.textContent = "Open Complete Route in Google Maps";
    }
    if (daySelect) {
      daySelect.innerHTML = routeMap.dayRoutes.map(function (route) {
        return '<option value="' + route.day + '">Day ' + route.day + " Route</option>";
      }).join("");
    }
    if (options.openDayRouteButton) {
      options.openDayRouteButton.textContent = "Open Day Route";
      options.openDayRouteButton.href = routeMap.dayRoutes[0] ? routeMap.dayRoutes[0].googleMapsDirectionsUrl : (routeMap.googleMapsDirectionsUrl || "#");
    }

    function syncSelectedDayLink() {
      if (!options.openDayRouteButton || !daySelect) return;
      var selectedDay = Number(daySelect.value || 1);
      var matchedRoute = routeMap.dayRoutes.find(function (route) {
        return route.day === selectedDay;
      }) || routeMap.dayRoutes[0];
      options.openDayRouteButton.href = matchedRoute ? matchedRoute.googleMapsDirectionsUrl : (routeMap.googleMapsDirectionsUrl || "#");
    }

    if (daySelect) {
      daySelect.onchange = syncSelectedDayLink;
      syncSelectedDayLink();
    }

    if (options.fallbackFrame) {
      options.fallbackFrame.src = "https://www.google.com/maps?q=" + encodeURIComponent(routeMap.destination || routeMap.endPoint || routeMap.origin || "") + "&output=embed";
    }

    return ensureMapsLoaded().then(function () {
      if (!options.mapElement) {
        return routeMap;
      }

      updateMessage(options, null);
      toggleFallback(options, false);
      options.mapElement.innerHTML = "";
      var map = new window.google.maps.Map(options.mapElement, {
        zoom: 7,
        center: { lat: 28.6139, lng: 77.2090 },
        mapTypeControl: false,
        streetViewControl: false,
        fullscreenControl: true
      });
      var bounds = new window.google.maps.LatLngBounds();
      var directionsService = new window.google.maps.DirectionsService();
      var renderers = [];

      return Promise.all(routeMap.dayRoutes.map(function (dayRoute, index) {
        return requestDirections(directionsService, dayRoute).then(function (result) {
          var renderer = new window.google.maps.DirectionsRenderer({
            map: map,
            suppressMarkers: true,
            preserveViewport: true,
            polylineOptions: {
              strokeColor: dayRoute.color,
              strokeOpacity: 0.9,
              strokeWeight: 6
            }
          });

          renderer.setDirections(result);
          renderers.push(renderer);

          result.routes[0].overview_path.forEach(function (point) {
            bounds.extend(point);
          });
        }).catch(function (error) {
          console.log("HKE route map day render failed:", error.message || error);
          if (index === 0) {
            throw error;
          }
        });
      })).then(function () {
        addRouteMarkers(map, routeMap);
        if (!bounds.isEmpty()) {
          map.fitBounds(bounds, 48);
        }
        options.mapElement.dataset.hkeRendered = "true";
        return routeMap;
      });
    }).catch(function (error) {
      console.log("HKE route map fallback:", error && (error.userMessage || error.message) || error);
      updateMessage(options, error || mapsLoadFailure);
      toggleFallback(options, true);
      return routeMap;
    });
  }

  window.HKERouteMap = {
    normalizeRouteMap: normalizeRouteMap,
    render: renderRouteMap
  };
})();
