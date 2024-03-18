return (() => {
  const COOKIE_PAGE_KEYWORDS = [
    "cookie",
    "Cookie",
    "COOKIE",
    "cookies",
    "Cookies",
    "COOKIES",
    "GDPR",
    "gdpr",
    "Gdpr",
    "Gdpr",
  ];
  const ACCEPT_COOKIES_KEYWORDS = [
    "accept",
    "Accept",
    "ACCEPT",
    "Allow",
    "allow",
    "accept all",
    "Accept all",
    "ACCEPT ALL",
    "Allow all",
    "allow all",

    // German
    "Akzeptieren",
    "akzeptieren",
    "Erlauben",
    "erlauben",
    "Zustimmen",
    "zustimmen",
    "Alle akzeptieren",
    "Alle erlauben",
    "Alle zustimmen",
    "Alle akzeptieren",

    // French
    "Accepter",
    "accepter",
    "Autoriser",
    "autoriser",
    "Tout accepter",
    "Tout autoriser",
    "Tout accepter",
    "Tout autoriser",

    // Spanish
    "Aceptar",
    "aceptar",
    "Permitir",
    "permitir",
    "Aceptar todo",
    "Aceptar todo",
    "Permitir todo",
    "permitir todo",

    // Italian
    "Accetta",
    "accetta",
    "Consenti",
    "consenti",
    "Accetta tutto",
    "accetta tutto",
    "Consenti tutto",
    "consenti tutto",

    // Dutch
    "Accepteren",
    "accepteren",
    "Toestaan",
    "toestaan",
    "Accepteer alles",
    "accepteer alles",
    "Sta alles toe",
    "sta alles toe",

    // Polish
    "Akceptuj",
    "akceptuj",
    "Zezwalaj",
    "zezwalaj",
    "Akceptuj wszystko",
    "akceptuj wszystko",
    "Zezwalaj na wszystko",
    "zezwalaj na wszystko",

    // Portuguese
    "Aceitar",
    "aceitar",
    "Permitir",
    "permitir",
    "Aceitar tudo",
    "aceitar tudo",
    "Permitir tudo",
    "permitir tudo",

    // Russian
    "Принять",
    "принять",
    "Разрешить",
    "разрешить",
    "Принять все",
    "принять все",
    "Разрешить все",
    "разрешить все",

    // Turkish
    "Kabul",
    "kabul",
    "İzin",
    "izin",
    "Hepsini kabul et",
    "hepsini kabul et",
    "Hepsine izin ver",
    "hepsine izin ver",

    // Czech
    "Přijmout",
    "přijmout",
    "Povolit",
    "povolit",
    "Přijmout vše",
    "přijmout vše",
    "Povolit vše",
    "povolit vše",

    // Swedish
    "Acceptera",
    "acceptera",
    "Tillåt",
    "tillåt",
    "Acceptera alla",
    "acceptera alla",
    "Tillåt alla",
    "tillåt alla",

    // Arabic
    "قبول",
    "قبول",
    "السماح",
    "السماح",
    "قبول الكل",
    "قبول الكل",
    "السماح بالكل",
    "السماح بالكل",
  ];

  function _is_privacy_gate(node) {
    if (!node) return false;

    let text = node.innerText;

    if (!text) return false;

    for (var keyword of COOKIE_PAGE_KEYWORDS) {
      if (text.search(keyword) != -1) return true;
    }
    return false;
  }

  function _check_if_in_gate(node) {
    if (!node) return null;

    if (_is_privacy_gate(node)) return node;

    if (node.parentElement == document.body) return null;

    return _check_if_in_gate(node.parentElement);
  }

  function _get_button_from_node(node) {
    let button = null;

    node.querySelectorAll('button, a[role="button"]').forEach((element) => {
      for (var keyword of ACCEPT_COOKIES_KEYWORDS) {
        if (button) break;

        if (element.innerText.search(keyword) == 0) {
          button = element;
          break;
        }
      }
    });

    return button;
  }

  function _get_button_from_ancestors(node) {
    let button = _get_button_from_node(node);
    if (button) return button;

    if (node.parentElement == document.body) return null;
    if (!node.parentElement) return null;

    return _get_button_from_ancestors(node.parentElement);
  }

  function query_screen_gated(checks = 3) {
    let points = [];

    // for (var i = 0; i < checks; i++) {
    //     points.push(
    //         { x: Math.random() * window.innerWidth, y: Math.random() * window.innerHeight }
    //     )
    // }

    points.push({ x: window.innerWidth / 2, y: window.innerHeight / 2 });

    var gate;
    var iframes = [];

    for (var point of points) {
      let node = document.elementFromPoint(point.x, point.y);

      if (!node) continue;

      if (node.nodeName == "IFRAME") {
        if (iframes.find((x) => x == node) == undefined) iframes.push(node);
        continue;
      }

      gate = _check_if_in_gate(node);
      if (gate) break;
    }

    if (!gate) return { iframes: iframes, button: null };

    //get the accept button

    var button = _get_button_from_ancestors(gate);

    return { iframes: iframes, button: button };
  }

  return query_screen_gated();
})();
