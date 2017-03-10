
var json;
var destinationType; // sets the format of returned value

// Wait for PhoneGap to connect with the device
//
document.addEventListener("deviceready", onDeviceReady, false);

// PhoneGap is ready to be used!
//
function onDeviceReady() {
    destinationType = navigator.camera.DestinationType;
}

function win(r) {
    hideSpinner();
    json = r.response;
    drawBoxes();
    console.log("Code = " + r.responseCode);
    console.log("Response = " + r.response);
    console.log("Sent = " + r.bytesSent);
}

function fail(evt) {
    hideSpinner();
    alert('Something went wrong. Check you are connected to the internet.')
    alert('fail ' + evt.target.error.code);
    console.log(evt.target.error.code);
}

function postData(data) {
    showSpinner();
    var options = new FileUploadOptions();
    options.fileKey = "image";
    options.mimeType = "image/jpeg";
    var ft = new FileTransfer();
    ft.upload(data, encodeURI('http://api.pyimagesearch.com/face_detection/detect/'), win, fail, options);
};

function drawBoxes() {

    if (json === undefined) return;
    
    var data = JSON.parse(json);
    if (!data.success || data.faces.length === 0) {
        alert('No faces found');
        alert(json);
        return;
    }

    // original dimensions
    var nwidth = document.getElementById('image').naturalWidth;
    var nheight = document.getElementById('image').naturalHeight;

    // scaled dimensions
    var width = $('#image').width();
    var height = $('#image').height();

    // adjuest for center-block class on image
    var ml = $('#image').css('marginLeft');
    var _left = +ml.replace('px', '');
    var mt = $('#image').css('marginTop');
    var _top = +mt.replace('px', '');

    data.faces.forEach(function (face) {

        var x0 = face[0];
        var y0 = face[1];
        var x1 = face[2];
        var y1 = face[3];

        var _x0 = (width / nwidth * x0 + _left).toFixed(0);
        var _y0 = (height / nheight * y0 + _top).toFixed(0);
        var _x1 = (width / nwidth * x1 + _left).toFixed(0) - _x0;
        var _y1 = (height / nheight * y1 + _top).toFixed(0) - _y0;

        console.info('style=\'left: '
            + _x0 + 'px; top: '
            + _y0 + 'px; width: '
            + _x1 + 'px; height: '
            + _y1 + 'px\'');

        jQuery('<div id=\'box\' style=\'left: '
            + _x0 + 'px; top: '
            + _y0 + 'px; width: '
            + _x1 + 'px; height: '
            + _y1 + 'px\' class=\'box img-responsive\'></div>', {
            }).appendTo('#wrapper');

    });
};

window.onload = drawBoxes;
window.onresize = drawBoxes;

// Called when a photo is successfully retrieved
//
function onPhotoFileSuccess(imageData) {

    // Remove any existing boxes
    $('.box').remove();

    // Display the image
    $('#image').attr("src", imageData);

    // Let's identify faces
    postData(imageData);
}

function capturePhotoWithFile() {
    navigator.camera.getPicture(onPhotoFileSuccess, onFail, { quality: 50, destinationType: Camera.DestinationType.FILE_URI });
}

// Called if something bad happens.
//
function onFail(message) {

    alert('Failed because: ' + message);

}

function showSpinner() {

    $('#camera').fadeOut(function () {
        $('#spinner').fadeIn();
    });

}

function hideSpinner() {

    $('#spinner').fadeOut(function () {
        $('#camera').fadeIn();
    });

}

$('#camera').click(function () {

    // Hide welcome page
    $('#landing').fadeOut(function () {

        // Take a picture
        capturePhotoWithFile();

    });

});
