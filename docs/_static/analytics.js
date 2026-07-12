(() => {
  const endpoint = "https://pyqed.org/api/analytics";
  const event = "docs_pageview";
  const payload = new Blob([event], { type: "text/plain;charset=UTF-8" });

  if (navigator.sendBeacon?.(endpoint, payload)) return;

  void fetch(endpoint, {
    method: "POST",
    body: event,
    headers: { "Content-Type": "text/plain;charset=UTF-8" },
    keepalive: true,
  });
})();
