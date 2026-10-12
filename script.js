(() => {
  'use strict';

  /* =====================================================================
   * CONFIGURATION
   * Change MODEL_URL / METADATA_URL if your model folder has another name.
   * Paths are relative to index.html. The weights file (weights.bin) is
   * NOT listed here: model.json names it in its "weightsManifest", and the
   * loader requests it relative to model.json's own location.
   * ===================================================================== */
  const CONFIG = {
    MODEL_URL: './model/model.json',
    METADATA_URL: './model/metadata.json',
    DEFAULT_THRESHOLD: 0.8,               // 0..1, adjustable with the slider
    MAX_FILE_BYTES: 10 * 1024 * 1024,     // 10 MB
    ACCEPTED_TYPES: ['image/jpeg', 'image/png', 'image/webp'],
    ACCEPTED_EXTENSIONS: ['.jpg', '.jpeg', '.png', '.webp'],
  };

  /* =====================================================================
   * UNKNOWN-OBJECT REJECTION
   * Each check receives the predictions sorted by probability (highest
   * first) plus a context object, and returns:
   *   { rejected: false }
   *   { rejected: true, reason: 'plain-language explanation' }
   * Checks run in order; the first one that rejects wins.
   *
   * To add a stronger method later (for example an entropy check, a
   * margin check between the top two classes, an "Other / Background"
   * class from retraining, or an out-of-distribution detector), write a
   * function with this shape and push it onto REJECTION_CHECKS.
   * ===================================================================== */
  function thresholdCheck(predictions, ctx) {
    const top = predictions[0];
    if (top.probability < ctx.threshold) {
      return {
        rejected: true,
        reason: `The highest score (${formatPercent(top.probability)}) is below the ${formatPercent(ctx.threshold)} threshold.`,
      };
    }
    return { rejected: false };
  }

  const REJECTION_CHECKS = [thresholdCheck];

  function evaluatePredictions(predictions, threshold) {
    const ctx = { threshold };
    for (const check of REJECTION_CHECKS) {
      const verdict = check(predictions, ctx);
      if (verdict && verdict.rejected) return verdict;
    }
    return { rejected: false };
  }

  /* =====================================================================
   * STATE + ELEMENTS
   * ===================================================================== */
  const state = {
    model: null,
    modelPromise: null,
    mode: 'upload',            // 'upload' | 'camera'
    busy: false,               // a prediction is running
    threshold: CONFIG.DEFAULT_THRESHOLD,
    lastPredictions: null,     // sorted predictions from the latest run
    // upload
    imageReady: false,
    objectUrl: null,
    // camera
    stream: null,
    cameraStarting: false,
    captured: false,
  };

  const $ = (id) => document.getElementById(id);
  const els = {
    modelStatus: $('modelStatus'),
    modelStatusText: $('modelStatusText'),
    retryLoad: $('retryLoad'),
    modelLoading: $('modelLoading'),
    modelError: $('modelError'),
    modelErrorText: $('modelErrorText'),

    tabUpload: $('tabUpload'),
    tabCamera: $('tabCamera'),
    panelUpload: $('panelUpload'),
    panelCamera: $('panelCamera'),

    dropZone: $('dropZone'),
    fileInput: $('fileInput'),
    previewStage: $('previewStage'),
    previewImg: $('previewImg'),
    uploadError: $('uploadError'),
    chooseAnotherBtn: $('chooseAnotherBtn'),
    clearBtn: $('clearBtn'),
    identifyUploadBtn: $('identifyUploadBtn'),

    facingSelect: $('facingSelect'),
    video: $('video'),
    captureCanvas: $('captureCanvas'),
    cameraPlaceholder: $('cameraPlaceholder'),
    cameraPlaceholderText: $('cameraPlaceholderText'),
    cameraError: $('cameraError'),
    startCameraBtn: $('startCameraBtn'),
    captureBtn: $('captureBtn'),
    identifyCameraBtn: $('identifyCameraBtn'),
    stopCameraBtn: $('stopCameraBtn'),

    stateEmpty: $('stateEmpty'),
    stateProcessing: $('stateProcessing'),
    stateError: $('stateError'),
    stateErrorText: $('stateErrorText'),
    outcome: $('outcome'),
    outcomeBadge: $('outcomeBadge'),
    outcomeLabel: $('outcomeLabel'),
    outcomeDetail: $('outcomeDetail'),
    bars: $('bars'),

    thresholdRange: $('thresholdRange'),
    thresholdOutput: $('thresholdOutput'),
  };

  const DEFAULT_PLACEHOLDER_HTML = els.cameraPlaceholderText.innerHTML;

  /* =====================================================================
   * HELPERS
   * ===================================================================== */
  function formatPercent(p) {
    return `${(p * 100).toFixed(1)}%`;
  }

  function nextPaint() {
    // Lets the browser paint the "Identifying…" state before heavy work starts.
    return new Promise((resolve) => requestAnimationFrame(() => setTimeout(resolve, 0)));
  }

  function showInlineError(el, message) {
    el.textContent = message;
    el.hidden = false;
  }
  function hideInlineError(el) {
    el.textContent = '';
    el.hidden = true;
  }

  /* =====================================================================
   * MODEL LOADING (once, then reused)
   * ===================================================================== */
  function setModelStatus(kind, text) {
    els.modelStatus.dataset.state = kind;
    els.modelStatusText.textContent = text;
    els.retryLoad.hidden = kind !== 'error';
  }

  /**
   * Checks the files before handing them to the loader so that failures
   * produce a specific, readable message instead of a vague one.
   */
  async function preflight(modelUrl, metadataUrl) {
    if (location.protocol === 'file:') {
      throw new Error(
        'This page was opened from the file system (file://). Browsers block model loading this way. ' +
        'Start a local server in the project folder and open http://localhost:8000 instead.'
      );
    }

    let modelRes;
    try {
      modelRes = await fetch(modelUrl, { cache: 'no-cache' });
    } catch (err) {
      throw new Error(`Could not request ${modelUrl}. Check your server and network connection.`);
    }
    if (!modelRes.ok) {
      throw new Error(`model.json was not found at ${modelUrl} (HTTP ${modelRes.status}). Check MODEL_URL in script.js and the folder name.`);
    }

    let modelJson;
    try {
      modelJson = await modelRes.json();
    } catch (err) {
      throw new Error(`${modelUrl} is not valid JSON. The file may be damaged or may be an HTML error page.`);
    }

    const manifest = modelJson && modelJson.weightsManifest;
    if (!Array.isArray(manifest) || manifest.length === 0) {
      throw new Error('model.json has no "weightsManifest", so the weight file location is unknown.');
    }

    // Weight files are resolved relative to model.json, exactly as the loader does it.
    const missing = [];
    for (const group of manifest) {
      for (const path of (group.paths || [])) {
        const weightUrl = new URL(path, modelUrl).href;
        try {
          const res = await fetch(weightUrl, { method: 'HEAD' });
          if (res.status === 404) missing.push(weightUrl);
        } catch (err) {
          // Network-level failure on a HEAD request: let the real load report it.
        }
      }
    }
    if (missing.length) {
      throw new Error(`model.json points to a weight file that was not found: ${missing.join(', ')}. Keep weights.bin in the same folder as model.json.`);
    }

    let metaRes;
    try {
      metaRes = await fetch(metadataUrl, { cache: 'no-cache' });
    } catch (err) {
      throw new Error(`Could not request ${metadataUrl}.`);
    }
    if (!metaRes.ok) {
      throw new Error(`metadata.json was not found at ${metadataUrl} (HTTP ${metaRes.status}). Check METADATA_URL in script.js.`);
    }
  }

  function loadModel() {
    if (state.model) return Promise.resolve(state.model);
    if (state.modelPromise) return state.modelPromise;

    state.modelPromise = (async () => {
      if (!window.tf || !window.tmImage) {
        throw new Error('The TensorFlow.js or Teachable Machine library did not load. Check your internet connection and reload.');
      }
      const modelUrl = new URL(CONFIG.MODEL_URL, document.baseURI).href;
      const metadataUrl = new URL(CONFIG.METADATA_URL, document.baseURI).href;

      await preflight(modelUrl, metadataUrl);

      const model = await window.tmImage.load(modelUrl, metadataUrl);
      const labels = model.getClassLabels();
      if (!labels || labels.length === 0) {
        throw new Error('The model loaded, but metadata.json lists no class labels.');
      }
      state.model = model;
      return model;
    })();

    // Allow a retry after failure.
    state.modelPromise.catch(() => { state.modelPromise = null; });
    return state.modelPromise;
  }

  async function initModel() {
    els.modelError.hidden = true;
    els.modelLoading.hidden = false;
    setModelStatus('loading', 'Loading model…');
    refreshControls();

    try {
      const model = await loadModel();
      const n = model.getTotalClasses();
      setModelStatus('ready', `Model ready · ${n} classes`);
      els.modelLoading.hidden = true;
    } catch (err) {
      console.error('Model load failed:', err);
      setModelStatus('error', 'Model not loaded');
      els.modelLoading.hidden = true;
      els.modelErrorText.textContent = err && err.message ? err.message : String(err);
      els.modelError.hidden = false;
    }
    refreshControls();
  }

  /* =====================================================================
   * CONTROLS: enable/disable according to state
   * ===================================================================== */
  function cameraActive() {
    return !!state.stream;
  }

  function refreshControls() {
    const ready = !!state.model;
    const busy = state.busy;

    els.identifyUploadBtn.disabled = busy || !ready || !state.imageReady;
    els.chooseAnotherBtn.disabled = busy;
    els.clearBtn.disabled = busy;

    const live = cameraActive();
    els.startCameraBtn.disabled = live || state.cameraStarting || busy;
    els.captureBtn.disabled = !live || busy;
    els.captureBtn.textContent = state.captured ? 'Retake' : 'Capture';
    els.identifyCameraBtn.disabled = busy || !ready || !live;
    els.stopCameraBtn.disabled = !live || busy;
    els.facingSelect.disabled = busy || state.cameraStarting;
  }

  /* =====================================================================
   * RESULT PANEL
   * ===================================================================== */
  function showResultState(name) {
    els.stateEmpty.hidden = name !== 'empty';
    els.stateProcessing.hidden = name !== 'processing';
    els.stateError.hidden = name !== 'error';
    els.outcome.hidden = name !== 'outcome';
  }

  function resetResult() {
    state.lastPredictions = null;
    showResultState('empty');
  }

  function renderError(message) {
    els.stateErrorText.textContent = message;
    showResultState('error');
  }

  function renderPredictions(predictions) {
    const verdict = evaluatePredictions(predictions, state.threshold);
    const top = predictions[0];

    els.outcome.dataset.kind = verdict.rejected ? 'uncertain' : 'success';

    if (verdict.rejected) {
      els.outcomeBadge.textContent = 'Uncertain';
      els.outcomeLabel.textContent = 'Object not confidently recognized';
      els.outcomeDetail.textContent =
        `${verdict.reason} The closest class was “${top.className}”, but do not treat it as a confirmed match.`;
    } else {
      els.outcomeBadge.textContent = 'Recognized';
      els.outcomeLabel.textContent = top.className;
      els.outcomeDetail.textContent = `Confidence: ${formatPercent(top.probability)}`;
    }

    els.bars.replaceChildren();
    predictions.forEach((p, i) => {
      const li = document.createElement('li');
      li.className = 'bar-row' + (i === 0 ? ' is-top' : '');

      const head = document.createElement('div');
      head.className = 'bar-head';
      const name = document.createElement('span');
      name.className = 'bar-name';
      name.textContent = p.className;
      const value = document.createElement('span');
      value.className = 'bar-value';
      value.textContent = formatPercent(p.probability);
      head.append(name, value);

      const track = document.createElement('div');
      track.className = 'bar-track';
      track.setAttribute('role', 'progressbar');
      track.setAttribute('aria-label', `${p.className} confidence`);
      track.setAttribute('aria-valuemin', '0');
      track.setAttribute('aria-valuemax', '100');
      track.setAttribute('aria-valuenow', (p.probability * 100).toFixed(1));
      const fill = document.createElement('div');
      fill.className = 'bar-fill';
      track.append(fill);

      li.append(head, track);
      els.bars.append(li);

      // Set width after insertion so the CSS transition animates.
      requestAnimationFrame(() => { fill.style.width = `${(p.probability * 100).toFixed(2)}%`; });
    });

    showResultState('outcome');
  }

  /* =====================================================================
   * PREDICTION (one at a time)
   * `source` may be an <img>, <canvas> or <video>. The Teachable Machine
   * library crops and resizes it to the model's input size itself.
   * ===================================================================== */
  async function runPrediction(source) {
    if (state.busy) return;
    if (!state.model) {
      renderError('The model is not loaded yet. Wait for “Model ready” in the header, or use Retry.');
      return;
    }

    state.busy = true;
    refreshControls();
    showResultState('processing');

    try {
      await nextPaint();
      const raw = await state.model.predict(source);
      const predictions = raw
        .map((p) => ({ className: p.className, probability: p.probability }))
        .sort((a, b) => b.probability - a.probability);

      state.lastPredictions = predictions;
      renderPredictions(predictions);
    } catch (err) {
      console.error('Prediction failed:', err);
      state.lastPredictions = null;
      renderError('The image could not be analyzed. Try a different image or capture again.');
    } finally {
      state.busy = false;
      refreshControls();
    }
  }

  /* =====================================================================
   * UPLOAD
   * ===================================================================== */
  function hasAcceptedExtension(name) {
    const lower = name.toLowerCase();
    return CONFIG.ACCEPTED_EXTENSIONS.some((ext) => lower.endsWith(ext));
  }

  function releaseObjectUrl() {
    if (state.objectUrl) {
      URL.revokeObjectURL(state.objectUrl);
      state.objectUrl = null;
    }
  }

  function clearImage() {
    releaseObjectUrl();
    els.previewImg.removeAttribute('src');
    els.previewStage.hidden = true;
    els.dropZone.hidden = false;
    els.chooseAnotherBtn.hidden = true;
    els.clearBtn.hidden = true;
    els.fileInput.value = '';
    state.imageReady = false;
    hideInlineError(els.uploadError);
    resetResult();
    refreshControls();
  }

  function handleFile(file) {
    hideInlineError(els.uploadError);
    if (!file) return;

    const typeOk = CONFIG.ACCEPTED_TYPES.includes(file.type) || (!file.type && hasAcceptedExtension(file.name));
    if (!typeOk) {
      showInlineError(els.uploadError, 'That file type is not supported. Choose a JPG, JPEG, PNG or WEBP image.');
      els.fileInput.value = '';
      return;
    }
    if (file.size === 0) {
      showInlineError(els.uploadError, 'That file is empty. Choose a different image.');
      els.fileInput.value = '';
      return;
    }
    if (file.size > CONFIG.MAX_FILE_BYTES) {
      const mb = Math.round(CONFIG.MAX_FILE_BYTES / (1024 * 1024));
      showInlineError(els.uploadError, `That image is larger than ${mb} MB. Choose a smaller file.`);
      els.fileInput.value = '';
      return;
    }

    const url = URL.createObjectURL(file);
    const probe = new Image();
    probe.onload = () => {
      if (!probe.naturalWidth || !probe.naturalHeight) {
        URL.revokeObjectURL(url);
        showInlineError(els.uploadError, 'That image has no readable pixels. Try another file.');
        return;
      }
      releaseObjectUrl();
      state.objectUrl = url;
      els.previewImg.src = url;
      els.previewStage.hidden = false;
      els.dropZone.hidden = true;
      els.chooseAnotherBtn.hidden = false;
      els.clearBtn.hidden = false;
      state.imageReady = true;
      resetResult();
      refreshControls();
    };
    probe.onerror = () => {
      URL.revokeObjectURL(url);
      showInlineError(els.uploadError, 'That image could not be read. It may be damaged, or your browser may not support this format. Try another file.');
      els.fileInput.value = '';
    };
    probe.src = url;
  }

  els.fileInput.addEventListener('change', () => handleFile(els.fileInput.files[0]));
  els.chooseAnotherBtn.addEventListener('click', () => {
    els.fileInput.value = '';
    els.fileInput.click();
  });
  els.clearBtn.addEventListener('click', clearImage);
  els.identifyUploadBtn.addEventListener('click', () => {
    if (!state.imageReady || !els.previewImg.complete) return;
    runPrediction(els.previewImg);
  });

  // Drag and drop
  ['dragenter', 'dragover'].forEach((evt) =>
    els.dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      els.dropZone.classList.add('is-over');
    })
  );
  ['dragleave', 'drop'].forEach((evt) =>
    els.dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      els.dropZone.classList.remove('is-over');
    })
  );
  els.dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer && e.dataTransfer.files;
    if (files && files.length) handleFile(files[0]);
  });

  /* =====================================================================
   * CAMERA
   * ===================================================================== */
  function cameraErrorMessage(err) {
    switch (err && err.name) {
      case 'NotAllowedError':
      case 'PermissionDeniedError':
        return 'Camera access was denied. Allow camera access for this site in your browser settings, then select Start camera again.';
      case 'NotFoundError':
      case 'DevicesNotFoundError':
        return 'No camera was found on this device.';
      case 'NotReadableError':
      case 'TrackStartError':
        return 'The camera is in use by another app or could not be started. Close other apps using the camera and try again.';
      case 'OverconstrainedError':
      case 'ConstraintNotSatisfiedError':
        return 'This device does not have the selected camera. Choose the other camera and try again.';
      case 'SecurityError':
        return 'The browser blocked camera access. Camera access needs a secure page (HTTPS or localhost).';
      case 'AbortError':
        return 'The camera could not be started. Try again.';
      default:
        return 'The camera could not be opened. Try again, or use the upload option instead.';
    }
  }

  function setCameraPlaceholder(html) {
    els.cameraPlaceholderText.innerHTML = html;
  }

  function showLiveVideo() {
    els.captureCanvas.hidden = true;
    els.video.hidden = false;
    els.cameraPlaceholder.hidden = true;
    state.captured = false;
  }

  function showCameraOff(messageHtml) {
    els.video.hidden = true;
    els.captureCanvas.hidden = true;
    els.cameraPlaceholder.hidden = false;
    setCameraPlaceholder(messageHtml || DEFAULT_PLACEHOLDER_HTML);
    state.captured = false;
  }

  function stopCamera(messageHtml) {
    if (state.stream) {
      state.stream.getTracks().forEach((track) => track.stop());
      state.stream = null;
    }
    els.video.srcObject = null;
    showCameraOff(messageHtml);
    refreshControls();
  }

  async function startCamera() {
    hideInlineError(els.cameraError);

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showInlineError(
        els.cameraError,
        window.isSecureContext
          ? 'This browser does not support camera access. Use the upload option instead.'
          : 'Camera access needs a secure page (HTTPS or localhost). Open the app from localhost or an HTTPS address.'
      );
      return;
    }
    if (state.cameraStarting) return;

    state.cameraStarting = true;
    stopCamera('Starting camera… allow access if your browser asks.');
    refreshControls();

    const facing = els.facingSelect.value;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: facing },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });

      // The user may have switched tabs or pressed Stop while permission was pending.
      if (state.mode !== 'camera') {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }

      state.stream = stream;
      els.video.srcObject = stream;
      await els.video.play();

      const settings = stream.getVideoTracks()[0].getSettings();
      const actualFacing = settings.facingMode || facing;
      els.video.classList.toggle('is-mirrored', actualFacing === 'user');
      els.captureCanvas.classList.toggle('is-mirrored', actualFacing === 'user');

      showLiveVideo();

      // If the camera disappears (unplugged, revoked), tidy up.
      stream.getVideoTracks()[0].addEventListener('ended', () => {
        if (state.stream === stream) {
          stopCamera('The camera stopped unexpectedly. Select <strong>Start camera</strong> to try again.');
        }
      });
    } catch (err) {
      console.error('Camera error:', err);
      stopCamera();
      showInlineError(els.cameraError, cameraErrorMessage(err));
    } finally {
      state.cameraStarting = false;
      refreshControls();
    }
  }

  /** Draws the current video frame onto the capture canvas at full resolution. */
  function grabFrame() {
    const v = els.video;
    if (!v.videoWidth || !v.videoHeight) return false;
    const c = els.captureCanvas;
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext('2d').drawImage(v, 0, 0, c.width, c.height);
    return true;
  }

  function toggleCapture() {
    hideInlineError(els.cameraError);
    if (!cameraActive()) return;

    if (state.captured) {
      showLiveVideo();            // Retake
      resetResult();
    } else {
      if (!grabFrame()) {
        showInlineError(els.cameraError, 'The camera is not ready yet. Wait a moment and try again.');
        return;
      }
      els.video.hidden = true;
      els.captureCanvas.hidden = false;
      state.captured = true;
      resetResult();
    }
    refreshControls();
  }

  function identifyFromCamera() {
    hideInlineError(els.cameraError);
    if (!cameraActive()) return;
    if (!state.captured) {
      if (!grabFrame()) {
        showInlineError(els.cameraError, 'The camera is not ready yet. Wait a moment and try again.');
        return;
      }
    }
    runPrediction(els.captureCanvas);
  }

  els.startCameraBtn.addEventListener('click', startCamera);
  els.stopCameraBtn.addEventListener('click', () => {
    hideInlineError(els.cameraError);
    stopCamera();
  });
  els.captureBtn.addEventListener('click', toggleCapture);
  els.identifyCameraBtn.addEventListener('click', identifyFromCamera);
  els.facingSelect.addEventListener('change', () => {
    if (cameraActive()) startCamera();   // restart with the newly selected camera
  });

  // Release the camera whenever the page no longer needs it.
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && cameraActive()) {
      stopCamera('The camera was stopped because the page was hidden. Select <strong>Start camera</strong> to resume.');
    }
  });
  window.addEventListener('pagehide', () => stopCamera());

  /* =====================================================================
   * TABS
   * ===================================================================== */
  function setMode(mode) {
    if (state.busy || mode === state.mode) return;
    state.mode = mode;
    const isUpload = mode === 'upload';

    els.tabUpload.setAttribute('aria-selected', String(isUpload));
    els.tabCamera.setAttribute('aria-selected', String(!isUpload));
    els.tabUpload.tabIndex = isUpload ? 0 : -1;
    els.tabCamera.tabIndex = isUpload ? -1 : 0;
    els.panelUpload.hidden = !isUpload;
    els.panelCamera.hidden = isUpload;

    if (isUpload) stopCamera();   // never leave the camera running in the background
    resetResult();
    refreshControls();
  }

  els.tabUpload.addEventListener('click', () => setMode('upload'));
  els.tabCamera.addEventListener('click', () => setMode('camera'));
  [els.tabUpload, els.tabCamera].forEach((tab) =>
    tab.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        const target = state.mode === 'upload' ? els.tabCamera : els.tabUpload;
        setMode(target === els.tabCamera ? 'camera' : 'upload');
        target.focus();
      }
    })
  );

  /* =====================================================================
   * THRESHOLD SLIDER
   * ===================================================================== */
  function applyThreshold(percent) {
    state.threshold = percent / 100;
    els.thresholdOutput.textContent = `${percent}%`;
    // Re-evaluate the latest result so the effect of the threshold is visible.
    if (state.lastPredictions && !state.busy) renderPredictions(state.lastPredictions);
  }

  els.thresholdRange.value = String(Math.round(CONFIG.DEFAULT_THRESHOLD * 100));
  els.thresholdRange.addEventListener('input', () => applyThreshold(Number(els.thresholdRange.value)));
  applyThreshold(Number(els.thresholdRange.value));

  /* =====================================================================
   * START
   * ===================================================================== */
  els.retryLoad.addEventListener('click', initModel);
  showResultState('empty');
  refreshControls();
  initModel();
})();
