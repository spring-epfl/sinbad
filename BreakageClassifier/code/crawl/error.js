(function () {
    if (window.logEvents) return;

    window.breakage_error_script_loaded = true;

    function captureUnhandled(errorMsg, url, lineNumber, column, errorObj) {
        var logMessage = {
            'timestamp': new Date(),
            'message': errorMsg,
            'src': url,
            'stack': errorObj.stack,
            'level': 'error'
        }
        window.logEvents.push(logMessage);
        window.last_log_event = logMessage;
    }

    function capture(level) {
        return function () {
            var args = Array.prototype.slice.call(arguments, 0);
            var logMessage = []
            var t = new Date()
            for (var i = 0; i < args.length; i++) {
                if (args[i] instanceof Error) {
                    let m = {
                        'timestamp': t,
                        'message': args[i].message,
                        'src': "",
                        'stack': args[i].stack,
                        'level': level
                    }
                    logMessage.push(m)
                    window.last_log_event = m;
                } else {
                    let m = {
                        'timestamp': t,
                        'message': args[i].message,
                        'src': "",
                        'stack': "",
                        'level': level
                    }
                    logMessage.push(m)
                    window.last_log_event = m;
                }
            }
            window.logEvents.concat(logMessage);
        }
    }
    console = console || {};
    //console.warn = capture('warn');
    console.error = capture('error');
    window.onerror = captureUnhandled;
    console.log("loaded error handler")
    window.logEvents = [];
    window.last_log_event = null;

    //var _body = document.body.innerHTML;
    //document.body.innerHTML = "";
    //setTimeout(() => { document.body.innerHTML = _body }, 500)
}());


