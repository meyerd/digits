var canvasDrawingApp = (function() {
    "use strict";

    var canvas, 
        context,
        canvasHeight = 100,
        canvasWidth = 100,
        clickX = [],
        clickY = [],
        clickDrag = [],
        redrawCallback = null,
        paint = false,

    getImageData = function() {
      var imgData = context.getImageData(0, 0, canvasWidth, canvasHeight);
    },

    setRedrawCallback = function(cb) {
      redrawCallback = cb;
    },

    clearCanvas = function() {
      context.clearRect(0, 0, canvasWidth, canvasHeight);
    },

    redraw = function() {
      clearCanvas();
      context.strokeStyle = "#000000";
      context.lineJoin = "round";
      context.lineWidth = 5;

      for(var i = 0; i < clickX.length; i++) {
        context.beginPath();
        if(clickDrag[i] && i) {
          context.moveTo(clickX[i - 1], clickY[i - 1]);
        } else {
          context.moveTo(clickX[i] - 1, clickY[i]);
        }
        context.lineTo(clickX[i], clickY[i]);
        context.closePath();
        context.stroke();
      }

      if(redrawCallback) {
        setTimeout(function() {
          redrawCallback(canvas);
        }, 1);
      }
    },

    addClick = function(x, y, dragging) {
      clickX.push(x);
      clickY.push(y);
      clickDrag.push(dragging);
    },

    createUserEvents = function() {
      var press = function(e) {
        var rect = canvas.getBoundingClientRect(),
        mouseX = (e.changedTouches ? e.changedTouches[0].clientX : e.clientX) - rect.left,
        mouseY = (e.changedTouches ? e.changedTouches[0].clientY : e.clientY) - rect.top;
        paint = true;
        addClick(mouseX, mouseY, false);
        redraw();
      },

      drag = function(e) {
         var rect = canvas.getBoundingClientRect(),
        mouseX = (e.changedTouches ? e.changedTouches[0].clientX : e.clientX) - rect.left,
        mouseY = (e.changedTouches ? e.changedTouches[0].clientY : e.clientY) - rect.top;

        if (paint) {
          addClick(mouseX, mouseY, true);
          redraw();
        }
        e.preventDefault();
      },

      release = function(e) {
        paint = false;
        redraw();
      },

      cancel = function(e) {
        paint = false;
      },

      clearButtonPress = function(e) {
        clickX = new Array();
        clickY = new Array();
        clickDrag = new Array();
        clearCanvas();
        document.getElementById('digit_output').innerText = "-";
      };

      canvas.addEventListener("mousedown", press, false);
      canvas.addEventListener("mousemove", drag, false);
      canvas.addEventListener("mouseup", release);
      canvas.addEventListener("mouseout", cancel, false);

      canvas.addEventListener("touchstart", press, false);
      canvas.addEventListener("touchmove", drag, false);
      canvas.addEventListener("touchend", release, false);
      canvas.addEventListener("touchcancel", cancel, false);
      var clearButton = document.getElementById("clear_canvas_button");
      clearButton.addEventListener("mousedown", clearButtonPress);
    },
    
    init = function() {
      canvas = document.createElement('canvas');
      canvas.setAttribute('width', canvasWidth);
      canvas.setAttribute('height', canvasHeight);
      canvas.setAttribute('id', 'canvas');
      document.getElementById('draw_canvas').appendChild(canvas);
      if (typeof G_vmlCanvasManager !== "undefined") {
        canvas = G_vmlCanvasManager.initElement(canvas);
      }
      context = canvas.getContext("2d");
      canvas.style.border = "1px solid black";
      redraw();
      createUserEvents();
    };

    return {
      init: init,
      setRedrawCallback: setRedrawCallback
    };
}());
