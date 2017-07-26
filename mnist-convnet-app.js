var convnetApp = (function() {
  "use strict";

	var layer_defs = [],
			net,
			classes_txt = ['0','1','2','3','4','5','6','7','8','9'],
			num_classes = 10,
			scaleCanvas,
			scaleCtx,
			digitdiv,
      confidencediv,
      debugCanvas,
      debugCtx,
      jsonlocation,

  loadJSON = function(callback, loc) {
    var xobj = new XMLHttpRequest();
    xobj.overrideMimeType("application/json");
    xobj.open('GET', loc, true);
    xobj.onreadystatechange = function() {
      if(xobj.readyState == 4 && xobj.status == "200") {
        callback(xobj.responseText);
      }
    };
    xobj.send(null);
  },
    	
	canvasCallback = function(imgDataCanvas) {
		// scale image to 28x28
		// var scale = 28 / imgDataCanvas.width;
		scaleCtx.clearRect(0, 0, 28, 28);
		// scaleCtx.scale(scale, scale);
		scaleCtx.drawImage(imgDataCanvas, 0, 0, 28, 28);
		var imgData = scaleCtx.getImageData(0, 0, 28, 28);
    debugCtx.clearRect(0, 0, 28, 28);
    var debugImgData = debugCtx.getImageData(0, 0, 28, 28);
		// load into vol
		var x = new convnetjs.Vol(1, 28, 28, 0.0);
		var W = 28*28;
		for(var i = 0; i < W; i++) {
      var g = imgData.data[i*4+3];

      debugImgData.data[i*4] = g;
      debugImgData.data[i*4+1] = g;
      debugImgData.data[i*4+2] = g;
      debugImgData.data[i*4+3] = 255.0;

			// x.w[i] = (255.0 - g) / 255.0;
			x.w[i] = (g) / 255.0;
		}
    debugCtx.putImageData(debugImgData, 0, 0);
		var a = net.forward(x);
		var preds = [];
		for(var k = 0; k < a.w.length; k++) {
			preds.push({k: k, p: a.w[k]});
		}
		preds.sort(function(a, b){return a.p<b.p ? 1: -1;});
		var class_txt = classes_txt[preds[0].k];

		digitdiv.innerText = class_txt;
    confidencediv.innerText = (preds[0].p * 100.0).toFixed(2);
	},

  parseJsonToNet = function(json) {
    var d = JSON.parse(json);
    console.log("got json weights, parsing ...");
    for(var i = 0; i < d.length; d++) {
      var a = d[i].arraypos;
      var filtersize = d[i].filtersize;
      var filters = d[i].filters;
      var biases = d[i].biases;
      for(var j = 0; j < biases.length; j++) {
        net.layers[a].biases.w[j] = biases[j];
      }
      for(var f = 0; f < filters.length; f++) {
        for(var j = 0; j < filtersize; j++) {
          net.layers[a].filters[f].w[j] = filters[f][j];
        }
      }
    }
  },

	init = function(cbf, jsloc) {
		layer_defs.push({type: 'input', out_sx: 28, out_sy: 28, out_depth: 1});
		layer_defs.push({type: 'conv', filters: 32, sx: 5, stride: 1, activation: 'relu'});
		layer_defs.push({type: 'pool', sx: 2, stride: 2});
		layer_defs.push({type: 'conv', filters: 32, sx: 5, stride: 1, activation: 'relu'});
		layer_defs.push({type: 'pool', sx: 2, stride: 2});
		layer_defs.push({type: 'fc', num_neurons: 256, activation: 'relu'});
		layer_defs.push({type: 'softmax', num_classes: 10});

		net = new convnetjs.Net();
		net.makeLayers(layer_defs);

		scaleCanvas = document.createElement('canvas');
		scaleCanvas.setAttribute('width', 28);
		scaleCanvas.setAttribute('height', 28);
		scaleCtx = scaleCanvas.getContext("2d");

    debugCanvas = document.createElement('canvas');
    debugCanvas.setAttribute('width', 28);
    debugCanvas.setAttribute('height', 28);
    debugCanvas.setAttribute('id', 'deb_canvas');
    document.getElementById('debug_canvas').appendChild(debugCanvas);
    if (typeof G_vmlCanvasManager !== "undefined") {
              canvas = G_vmlCanvasManager.initElement(debugCanvas);
            }
    debugCanvas.style.border = "1px solid black";
    debugCtx = debugCanvas.getContext("2d");

		digitdiv = document.getElementById('digit_output');
    confidencediv = document.getElementById('digit_confidence');

    jsonlocation = jsloc;

    loadJSON(parseJsonToNet, jsonlocation);

		cbf.setRedrawCallback(canvasCallback);
	};

	return {
		init: init
	};
}());
