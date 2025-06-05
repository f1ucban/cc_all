class FRS {
    constructor() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.datetimeElement = document.getElementById('datetime');
        this.dateElement = document.getElementById('date');
        this.timeElement = document.getElementById('time');
        this.faceCountElement = document.getElementById('face-count');
        this.userList = document.getElementById('user-list');
        this.userCount = document.getElementById('user-count');
        this.searchInput = document.getElementById('search-user');
        this.recognitionLogs = document.getElementById('recognition-logs');

        this.enrollVideo = document.getElementById('enroll-video');
        this.enrollCanvas = document.getElementById('enroll-canvas');
        this.enrollCtx = this.enrollCanvas.getContext('2d');

        this.recognitionInterval = 5;
        this.frameCount = 0;
        this.lastTimestamp = 0;
        this.fps = 0;
        this.knownFaces = {};
        this.currentStream = null;
        this.enrollStream = null;
        this.isFrontCamera = true;
        this.isProcessing = false;
        this.lastFaces = [];
        this.lastLogTimes = {};
        this.logCooldown = 5 * 60 * 1000;
        this.minConfidence = 0.9;
        this.captureCanvas = document.getElementById('capture-canvas');
        this.captureCtx = this.captureCanvas.getContext('2d');
        this.isEnrollingGait = false;
        this.enrollGaitBtn = document.getElementById('enroll-gait-btn');
        this.captureGaitBtn = document.getElementById('capture-gait-btn');

        this.gaitEnrollVideo = document.getElementById('gait-enroll-video');
        this.gaitEnrollCanvas = document.getElementById('gait-enroll-canvas');
        this.gaitEnrollCtx = this.gaitEnrollCanvas.getContext('2d');
        this.isRecordingGait = false;
        this.gaitFrames = [];
        this.gaitStream = null;
        this.recordingInterval = null;
        this.recordingDuration = 5000;
        this.recordingStartTime = 0;
        this.gaitVideoFile = null;

        this.logsRecordsBtn = document.getElementById('logs-records-btn');
        this.detectionHistory = {};
        this.processFrameRunning = false;
        this.debug = true;

        this.gaitEnrollEventsSetup = false;
        this.recognitionHistory = {};
        this.minRecognitionDuration = 2000;
        this.recognitionThreshold = 0.9;
        this.gaitRecognitionThreshold = 0.3;

        this.lastFrameTime = 0;
        this.targetFPS = 30;
        this.frameInterval = 1000 / this.targetFPS;

        this.lastGaitMatch = null;
        this.gaitMatchTimeout = 2000;
        this.lastGaitMatchTime = 0;

        this.init();
        this.updateDateTime();
        setInterval(() => this.updateDateTime(), 1000);
    }

    async init() {
        try {
            await this.startCamera();
            const users = await this.loadUserList();
            this.renderUserList(users);
            this.setupEventListeners();
            this.setupSidebarToggles();
            this.setupEnrollmentPanelEvents();
            this.startStreaming();

            const welcomeLog = document.createElement('li');
            welcomeLog.className = 'log-entry success fade-in';
            welcomeLog.innerHTML = `
                <span class="log-timestamp">${new Date().toLocaleTimeString()}</span>
                <span class="log-message">System Initialized</span>
            `;
            this.recognitionLogs.prepend(welcomeLog);
        } catch (error) {
            console.error("Initialization error:", error);
            this.showLogMessage("System initialization failed: " + error.message, "error");
        }
    }

    setupEventListeners() {
        document.getElementById('toggle-camera').addEventListener('click', () => {
            this.isFrontCamera = !this.isFrontCamera;
            this.startCamera().catch(console.error);
        });

        document.getElementById('refresh-users').addEventListener('click', async () => {
            try {
                const users = await this.loadUserList();
                this.renderUserList(users);
                this.showLogMessage('User list refreshed', 'success');
            } catch (error) {
                console.error('Error refreshing users:', error);
                this.showLogMessage('Failed to refresh users', 'error');
            }
        });

        document.getElementById('clear-logs').addEventListener('click', () => {
            this.recognitionLogs.innerHTML = '';
            this.showLogMessage("Logs cleared", "info");
        });

        this.searchInput.addEventListener('input', () => {
            const term = this.searchInput.value.toLowerCase();
            document.querySelectorAll('.user-item').forEach(item => {
                item.style.display = item.dataset.searchTerm.includes(term) ? 'flex' : 'none';
            });
        });

        this.logsRecordsBtn.addEventListener('click', () => {
            this.showLogsRecords();
        });
    }

    async startCamera() {
        try {
            const constraints = {
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: this.isFrontCamera ? 'user' : 'environment'
                }
            };

            if (this.currentStream) {
                this.currentStream.getTracks().forEach(track => track.stop());
            }

            this.currentStream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video.srcObject = this.currentStream;

            return new Promise((resolve) => {
                this.video.onloadedmetadata = () => {
                    this.canvas.width = this.video.videoWidth;
                    this.canvas.height = this.video.videoHeight;
                    resolve();
                };
            });
        } catch (err) {
            console.error("Camera error:", err);
            this.showLogMessage(`Camera Error: ${err.message}`, "error");
            throw err;
        }
    }

    setupSidebarToggles() {
        document.getElementById('toggle-logs').addEventListener('click', () => {
            document.querySelector('.logs-section').classList.add('active');
            document.querySelector('.users-section').classList.remove('active');
            document.getElementById('toggle-logs').classList.add('active');
            document.getElementById('toggle-users').classList.remove('active');
        });

        document.getElementById('toggle-users').addEventListener('click', () => {
            document.querySelector('.users-section').classList.add('active');
            document.querySelector('.logs-section').classList.remove('active');
            document.getElementById('toggle-users').classList.add('active');
            document.getElementById('toggle-logs').classList.remove('active');
        });
    }

    async processFrame() {
        if (this.video.paused || this.video.ended) {
            return requestAnimationFrame(() => this.processFrame());
        }

        const currentTime = performance.now();
        const elapsed = currentTime - this.lastFrameTime;

        if (elapsed > this.frameInterval) {
            try {
                // Draw the video frame
                this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);

                // Draw face boxes if available
                if (this.lastFaces && this.lastFaces.length > 0) {
                    this.drawFaceBoxes(this.lastFaces);
                }

                this.lastFrameTime = currentTime;
            } catch (error) {
                console.error('Error in processFrame:', error);
                this.processFrameRunning = false;
            }
        }

        requestAnimationFrame(() => this.processFrame());
    }

    drawFaceBoxes(faces) {
        if (!faces || faces.length === 0) return;

        faces.forEach((face, index) => {
            const { x, y, w, h, name, confidence, type } = face;

            let boxColor;
            if (name && name.toLowerCase() === 'unknown') {
                boxColor = '#ff0000'; // Red for unknown faces
            } else if (type === 'face' && confidence >= this.recognitionThreshold) {
                boxColor = '#00ff00'; // Green for high confidence face match
            } else if (type === 'gait' && confidence >= this.gaitRecognitionThreshold) {
                boxColor = '#0000ff'; // Blue for high confidence gait match
            } else if (type === 'fusion' && confidence >= this.recognitionThreshold) {
                boxColor = '#ff00ff'; // Magenta for fusion match
            } else {
                boxColor = '#ff0000'; // Red for low confidence
            }

            this.ctx.beginPath();
            this.ctx.strokeStyle = boxColor;
            this.ctx.lineWidth = 3;
            this.ctx.strokeRect(x, y, w, h);

            let labelText = name && name.toLowerCase() !== 'unknown' ? name : 'Unknown';

            const labelWidth = this.ctx.measureText(labelText).width + 10;
            this.ctx.fillStyle = boxColor;
            this.ctx.fillRect(x, y - 20, labelWidth, 20);

            // Draw label text
            this.ctx.fillStyle = '#ffffff';
            this.ctx.font = '15px Arial';
            this.ctx.fillText(labelText, x + 5, y - 5);
        });
    }

    startStreaming() {
        console.log('Starting video stream...');
        this.video.addEventListener('play', () => {
            console.log('Video started playing, setting up streaming interval');
            this.streamingInterval = setInterval(async () => {
                try {
                    this.captureCanvas.width = this.video.videoWidth;
                    this.captureCanvas.height = this.video.videoHeight;
                    this.captureCtx.drawImage(this.video, 0, 0, this.captureCanvas.width, this.captureCanvas.height);

                    const blob = await new Promise(resolve => {
                        this.captureCanvas.toBlob(resolve, 'image/jpeg', 0.7);
                    });

                    if (blob) {
                        const formData = new FormData();
                        formData.append('frame', blob);

                        const response = await fetch('/process_frame', {
                            method: 'POST',
                            body: formData
                        });

                        if (response.ok) {
                            const data = await response.json();
                            this.handleRecognitionResults(data);
                        } else {
                            console.error('Failed to process frame:', await response.text());
                        }
                    }
                } catch (error) {
                    console.error('Error in streaming loop:', error);
                }
            }, 200);
        });

        if (this.video.readyState >= 3) {
            console.log('Video already ready, dispatching play event');
            this.video.dispatchEvent(new Event('play'));
        }
    }

    handleRecognitionResults(data) {
        if (!data) return;

        const now = new Date();
        this.cleanupRecognitionHistory();
        this.faceCountElement.textContent = data.face_matches ? data.face_matches.length : 0;

        let currentFaces = [];
        if (data.face_matches && data.face_matches.length > 0) {
            data.face_matches.forEach(face => {
                if (face.bbox) {
                    const [x, y, w, h] = face.bbox;
                    currentFaces.push({
                        x, y, w, h,
                        name: face.identity,
                        confidence: face.confidence,
                        type: 'face'
                    });
                }
                // Log face recognition results
                this.processRecognitionLog({
                    identity: face.identity,
                    confidence: face.confidence,
                    modality: 'face',
                    role: face.role
                }, now);
            });
        }

        // Handle gait recognition results
        if (data.gait_match) {
            this.lastGaitMatch = data.gait_match;
            this.lastGaitMatchTime = Date.now();

            if (data.gait_match.bbox) {
                const [x, y, w, h] = data.gait_match.bbox;
                currentFaces.push({
                    x, y, w, h,
                    name: data.gait_match.firstname + ' ' + data.gait_match.lastname,
                    confidence: data.gait_match.confidence,
                    type: 'gait'
                });
            }

            this.processRecognitionLog({
                identity: data.gait_match.firstname + ' ' + data.gait_match.lastname,
                confidence: data.gait_match.confidence,
                modality: 'gait',
                role: data.gait_match.role
            }, now);
        } else if (this.lastGaitMatch && (Date.now() - this.lastGaitMatchTime) < this.gaitMatchTimeout) {
            if (this.lastGaitMatch.bbox) {
                const [x, y, w, h] = this.lastGaitMatch.bbox;
                currentFaces.push({
                    x, y, w, h,
                    name: this.lastGaitMatch.firstname + ' ' + this.lastGaitMatch.lastname,
                    confidence: this.lastGaitMatch.confidence,
                    type: 'gait'
                });
            }
        }

        if (data.fusion_match) {
            if (data.fusion_match.bbox) {
                const [x, y, w, h] = data.fusion_match.bbox;
                currentFaces.push({
                    x, y, w, h,
                    name: data.fusion_match.identity,
                    confidence: data.fusion_match.confidence,
                    type: 'fusion'
                });
            }

            this.processRecognitionLog({
                identity: data.fusion_match.identity,
                confidence: data.fusion_match.confidence,
                modality: 'fusion',
                role: data.fusion_match.role
            }, now);
        }

        this.lastFaces = currentFaces;
        if (!this.processFrameRunning) {
            this.processFrameRunning = true;
            this.processFrame();
        }
    }

    processRecognitionLog(recognition, now) {
        console.log('Processing recognition log:', recognition);
        const { identity, confidence, modality } = recognition;

        if (!identity || identity.toLowerCase() === 'unknown') {
            console.log('Skipping log - identity is unknown');
            return;
        }

        let threshold = this.recognitionThreshold;
        if (modality === 'gait') {
            threshold = this.gaitRecognitionThreshold;
            console.log(`Using gait threshold: ${threshold}, confidence: ${confidence}`);
        }
        if (confidence < threshold) {
            console.log(`Skipping log - confidence ${confidence} below threshold ${threshold}`);
            return;
        }

        if (!this.recognitionHistory[identity]) {
            this.recognitionHistory[identity] = {
                firstSeen: now,
                lastSeen: now,
                count: 1
            };
        } else {
            this.recognitionHistory[identity].lastSeen = now;
            this.recognitionHistory[identity].count++;
        }


        const lastLogTime = this.lastLogTimes[identity] || 0;
        const timeSinceLastLog = now - lastLogTime;
        if (timeSinceLastLog < this.logCooldown) {
            console.log(`Skipping log - cooldown period not elapsed for ${identity}`);
            return;
        }

        const recognitionDuration = now - this.recognitionHistory[identity].firstSeen;
        if (recognitionDuration < this.minRecognitionDuration) {
            console.log(`Skipping log - recognition duration too short for ${identity}`);
            return;
        }

        console.log(`Logging recognition for ${identity} after ${recognitionDuration}ms of stable recognition`);
        this.lastLogTimes[identity] = now;
        this.addRecognitionLog(recognition);

        this.recognitionHistory[identity] = {
            firstSeen: now,
            lastSeen: now,
            count: 1
        };
    }

    cleanupRecognitionHistory() {
        const now = Date.now();
        const maxHistoryAge = 10000;

        Object.keys(this.recognitionHistory).forEach(identity => {
            const history = this.recognitionHistory[identity];
            if (now - history.lastSeen > maxHistoryAge) {
                console.log(`Cleaning up old recognition history for ${identity}`);
                delete this.recognitionHistory[identity];
            }
        });
    }

    addRecognitionLog(recognition) {
        if (recognition.identity === 'Unknown') {
            console.log('Skipping log for Unknown identity');
            return;
        }
        console.log('=== START addRecognitionLog ===');
        console.log('Adding log for:', recognition);
        const now = new Date();
        const timeString = now.toLocaleTimeString();
        const logItem = document.createElement('li');
        logItem.className = 'log-item fade-in';

        let modalityLabel = 'FACE';
        if (recognition.modality === 'gait') {
            modalityLabel = 'GAIT';
        } else if (recognition.modality === 'fusion' || recognition.modality === 'fused') {
            modalityLabel = 'FUSION';
        }
        logItem.innerHTML = `
            <div class="log-row log-row-main">
                <span class="log-timestamp">${timeString}</span>
                <span class="log-identity">${recognition.identity || recognition.firstname + ' ' + recognition.lastname || 'Unknown'}</span>
                <span class="log-modality">${modalityLabel}</span>
                <span class="log-confidence">${(recognition.confidence * 100).toFixed(1)}%</span>
            </div>
            ${recognition.role ? `<div class="log-row log-row-role"><span class="log-role">${recognition.role}</span></div>` : ''}
        `;

        this.recognitionLogs.prepend(logItem);

        if (this.recognitionLogs.children.length > 50) {
            this.recognitionLogs.removeChild(this.recognitionLogs.lastChild);
        }
        console.log('=== END addRecognitionLog ===');
    }

    showToastNotification(message) {
        const toast = document.createElement('div');
        toast.className = 'toast-notification fade-in';
        toast.textContent = message;
        document.body.appendChild(toast);

        setTimeout(() => {
            toast.classList.add('fade-out');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    showLogMessage(message, type = "info") {
        if (type === "error" || type === "success") {
            const logEntry = document.createElement('li');
            logEntry.className = `log-entry ${type}`;

            const timestamp = new Date().toLocaleTimeString();
            const messageSpan = document.createElement('span');
            messageSpan.textContent = message;

            logEntry.innerHTML = `
                <span class="log-time">${timestamp}</span>
                <span class="log-message">${message}</span>
            `;

            this.recognitionLogs.insertBefore(logEntry, this.recognitionLogs.firstChild);

            // Keep only last 50 logs
            while (this.recognitionLogs.children.length > 50) {
                this.recognitionLogs.removeChild(this.recognitionLogs.lastChild);
            }
        }
    }

    async loadUserList() {
        try {
            const response = await fetch('/users');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            if (!data || !data.users) {
                throw new Error('Invalid response format');
            }
            return data.users;
        } catch (error) {
            console.error('Error loading user list:', error);
            this.showLogMessage('Failed to load user list: ' + error.message, 'error');
            return [];
        }
    }

    renderUserList(users) {
        this.userList.innerHTML = '';
        this.knownFaces = {};

        if (users.length > 0) {
            this.userCount.textContent = users.length;
            users.forEach(user => {
                const userId = user.user_idx;
                const fullName = `${user.firstname} ${user.lastname}`;
                this.knownFaces[userId] = fullName;

                const li = document.createElement('li');
                li.className = 'user-item';
                li.dataset.userId = userId;
                li.dataset.searchTerm = `${user.firstname.toLowerCase()} ${user.lastname.toLowerCase()} ${user.role ? user.role.toLowerCase() : ''}`;

                li.innerHTML = `
                    <div class="user-info">
                        <div class="user-name">${fullName}</div>
                        ${user.role ? `<div class="user-role">${user.role}</div>` : ''}
                    </div>
                    <button class="remove-btn" data-user-id="${userId}">
                        <i class="fas fa-trash-alt"></i>
                    </button>
                `;
                this.userList.appendChild(li);
            });

            document.querySelectorAll('.remove-btn').forEach(button => {
                button.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.removeUser(e.currentTarget.dataset.userId);
                });
            });
        } else {
            this.userList.innerHTML = '<li class="no-users">No users enrolled yet</li>';
            this.userCount.textContent = '0';
        }
    }

    async removeUser(userId) {
        try {
            const user = this.knownFaces[userId] || `ID: ${userId}`;
            const { isConfirmed } = await Swal.fire({
                title: 'Confirm Removal',
                text: `Remove ${user}?`,
                icon: 'warning',
                showCancelButton: true,
                confirmButtonColor: '#d33',
                cancelButtonColor: '#3085d6',
                confirmButtonText: 'Yes, remove',
                customClass: {
                    popup: 'remove-confirm-modal'
                }
            });

            if (!isConfirmed) return;

            const response = await fetch(`/remove_user/${userId}`, { method: 'DELETE' });
            if (!response.ok) throw new Error(await response.json().error || 'Failed to remove user');

            this.showLogMessage(`User ${user} removed`, 'success');
            // Refresh user list immediately after successful deletion
            const users = await this.loadUserList();
            this.renderUserList(users);
        } catch (error) {
            console.error('Error removing user:', error);
            this.showLogMessage(`Error: ${error.message}`, 'error');
        }
    }

    async startEnrollmentCamera() {
        try {
            const constraints = {
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                }
            };

            if (this.enrollStream) {
                this.enrollStream.getTracks().forEach(track => track.stop());
            }

            this.enrollStream = await navigator.mediaDevices.getUserMedia(constraints);
            this.enrollVideo.srcObject = this.enrollStream;

            return new Promise((resolve) => {
                this.enrollVideo.onloadedmetadata = () => {
                    this.enrollCanvas.width = this.enrollVideo.videoWidth;
                    this.enrollCanvas.height = this.enrollVideo.videoHeight;
                    resolve();
                };
            });
        } catch (err) {
            console.error("Enrollment camera error:", err);
            this.showLogMessage(`Camera Error: ${err.message}`, "error");
            throw err;
        }
    }

    showEnrollmentPanel() {
        this.hideGaitEnrollmentPanel();

        this.enrollImages = [];
        this.updateEnrollThumbnails();
        document.getElementById('enroll-firstname').value = '';
        document.getElementById('enroll-lastname').value = '';
        document.getElementById('enroll-role').value = '';
        document.getElementById('enroll-count').textContent = '0/10';
        document.getElementById('enroll-submit-btn').disabled = true;
        document.getElementById('enroll-panel').style.display = 'block';
        document.querySelector('.app-container').classList.add('enrolling');

        this.startEnrollmentCamera().catch(console.error);
    }

    hideEnrollmentPanel() {
        document.getElementById('enroll-panel').style.display = 'none';
        document.querySelector('.app-container').classList.remove('enrolling');
        if (this.enrollStream) {
            this.enrollStream.getTracks().forEach(track => track.stop());
            this.enrollStream = null;
        }
    }

    updateEnrollThumbnails() {
        const container = document.getElementById('enroll-thumbnails');
        container.innerHTML = '';
        this.enrollImages.forEach(blob => {
            const img = document.createElement('img');
            img.src = URL.createObjectURL(blob);
            img.className = 'enroll-thumb';
            container.appendChild(img);
        });
        document.getElementById('enroll-count').textContent = `${this.enrollImages.length}/10`;
        document.getElementById('enroll-submit-btn').disabled = this.enrollImages.length < 5;
        document.getElementById('capture-face-btn').disabled = this.enrollImages.length >= 10;
        document.getElementById('enroll-upload-btn').disabled = this.enrollImages.length >= 10;
        const uploadLabel = document.querySelector('.enroll-upload-label');
        if (uploadLabel) {
            uploadLabel.classList.toggle('disabled', this.enrollImages.length >= 10);
        }
    }

    setupEnrollmentPanelEvents() {
        document.getElementById('enroll-btn').addEventListener('click', () => this.showEnrollmentPanel());
        document.getElementById('enroll-close-btn').addEventListener('click', () => this.hideEnrollmentPanel());
        document.getElementById('enroll-cancel-btn').addEventListener('click', () => this.hideEnrollmentPanel());

        document.getElementById('capture-face-btn').addEventListener('click', () => {
            if (this.enrollImages.length >= 10) return;

            this.enrollCanvas.width = this.enrollVideo.videoWidth;
            this.enrollCanvas.height = this.enrollVideo.videoHeight;
            this.enrollCtx.drawImage(this.enrollVideo, 0, 0, this.enrollCanvas.width, this.enrollCanvas.height);

            this.enrollCanvas.toBlob(blob => {
                this.enrollImages.push(blob);
                this.updateEnrollThumbnails();
            }, 'image/jpeg', 0.9);
        });

        document.getElementById('enroll-upload-btn').addEventListener('change', (e) => {
            const files = Array.from(e.target.files);
            for (let i = 0; i < files.length && this.enrollImages.length < 10; i++) {
                this.enrollImages.push(files[i]);
            }
            this.updateEnrollThumbnails();
            e.target.value = '';
        });

        document.getElementById('enroll-thumbnails').addEventListener('click', (e) => {
            const thumb = e.target.closest('.enroll-thumb');
            if (thumb) {
                const index = Array.from(thumb.parentNode.children).indexOf(thumb);
                this.enrollImages.splice(index, 1);
                this.updateEnrollThumbnails();
            }
        });

        document.getElementById('enroll-submit-btn').addEventListener('click', async () => {
            const firstname = document.getElementById('enroll-firstname').value.trim();
            const lastname = document.getElementById('enroll-lastname').value.trim();
            const role = document.getElementById('enroll-role').value.trim();

            if (!firstname || !lastname || this.enrollImages.length < 5) {
                this.showLogMessage('Please provide name and 5 to 10 images', 'error');
                return;
            }

            const formData = new FormData();
            formData.append('firstname', firstname);
            formData.append('lastname', lastname);
            formData.append('role', role);
            this.enrollImages.forEach((img, idx) => {
                formData.append('images', img, `face${idx}.jpg`);
            });

            const submitBtn = document.getElementById('enroll-submit-btn');
            const originalBtnText = submitBtn.innerHTML;
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
            this.showLogMessage('Starting face enrollment process...', 'info');

            try {
                const response = await fetch('/enroll', { method: 'POST', body: formData });
                const result = await response.json();
                if (!response.ok) throw new Error(result.error || 'Enrollment failed');

                this.showLogMessage(`User ${firstname} ${lastname} enrolled`, 'success');
                this.hideEnrollmentPanel();
                // Refresh user list immediately after successful enrollment
                const users = await this.loadUserList();
                this.renderUserList(users);
            } catch (error) {
                this.showLogMessage(error.message, 'error');
            } finally {
                submitBtn.innerHTML = originalBtnText;
                submitBtn.disabled = false;
            }
        });

        document.getElementById('enroll-gait-btn').addEventListener('click', () => {
            this.showGaitEnrollmentPanel();
        });

        this.logsRecordsBtn.addEventListener('click', () => this.showLogsRecords());

        // Add gait enrollment file upload handler
        const gaitUploadBtn = document.getElementById('gait-enroll-upload-btn');
        if (gaitUploadBtn) {
            gaitUploadBtn.addEventListener('change', async (event) => {
                const files = event.target.files;
                if (files.length > 0) {
                    const file = files[0];
                    if (file.type.startsWith('video/')) {
                        // Handle video file
                        this.gaitVideoFile = file;
                        console.log('Video file selected:', file.name, file.size, 'bytes');

                        // Show filename preview
                        const filePreview = document.querySelector('.file-preview');
                        if (filePreview) {
                            filePreview.innerHTML = `<div class="filename">${file.name}</div>`;
                        }

                        // Process and display the video
                        await this.processGaitVideo(file);

                        // Enable submit button
                        const submitBtn = document.getElementById('gait-enroll-submit');
                        if (submitBtn) {
                            submitBtn.disabled = false;
                            console.log('Submit button enabled after video selection');
                        }
                    } else if (file.type.startsWith('image/')) {
                        // Handle image files
                        this.gaitFrames = [];
                        for (const file of files) {
                            const blob = await file.arrayBuffer();
                            this.gaitFrames.push(new Blob([blob], { type: 'image/jpeg' }));
                        }
                        console.log('Image files selected:', this.gaitFrames.length, 'files');

                        // Show filename preview
                        const filePreview = document.querySelector('.file-preview');
                        if (filePreview) {
                            filePreview.innerHTML = `<div class="filename">${files.length} image files selected</div>`;
                        }

                        // Enable submit button
                        const submitBtn = document.getElementById('gait-enroll-submit');
                        if (submitBtn) {
                            submitBtn.disabled = false;
                            console.log('Submit button enabled after image selection');
                        }
                    } else {
                        this.showLogMessage('Please select a video or image file', 'error');
                    }
                }
            });
        }
    }

    async showLogsRecords() {
        try {
            const { value: password } = await Swal.fire({
                title: 'Enter Password',
                input: 'password',
                inputLabel: 'Password',
                inputPlaceholder: 'Enter your password',
                showCancelButton: true,
                backdrop: true,
                customClass: {
                    popup: 'password-prompt-modal'
                },
                confirmButtonText: 'View Logs & Records',
                inputValidator: (value) => {
                    if (!value) {
                        return 'You need to enter a password!';
                    }
                }
            });

            if (password) {
                const formData = new FormData();
                formData.append('password', password);

                const response = await fetch('/get_logs_records', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok) {
                    if (!data.users || !data.logs) {
                        throw new Error('Invalid response format: missing users or logs data');
                    }
                    this.displayLogsRecords(data);
                } else {
                    throw new Error(data.error || 'Failed to fetch logs and records');
                }
            }
        } catch (error) {
            console.error('Error showing logs and records:', error);
            Swal.fire({
                title: 'Error',
                text: error.message || 'Failed to show logs and records',
                icon: 'error',
                confirmButtonText: 'OK'
            });
        }
    }

    displayLogsRecords(data) {
        const { users, logs } = data;
        const defaultAvatarUrl = '/static/default-avatar.png';

        let content = `
            <div class="logs-records-container">
                <div class="logs-records-section">
                    <h3>Recognition Logs</h3>
                    <div class="logs-list">
                        ${logs.length === 0 ? '<p>No recognition logs found.</p>' : logs.map(log => `
                            <div class="log-item">
                                <div class="log-image">
                                    <img src="${log.recog_img || defaultAvatarUrl}" alt="Recognition Image" onerror="this.src='${defaultAvatarUrl}'">
                                </div>
                                <div class="log-details">
                                    <div class="log-identity">${log.identity || 'Unknown'}</div>
                                    <div class="log-modality">${log.modality || 'Unknown'}</div>
                                    <div class="log-confidence">Confidence: ${((log.confidence || 0) * 100).toFixed(1)}%</div>
                                    <div class="log-timestamp">${new Date(log.timestamp).toLocaleString()}</div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>

                <div class="logs-records-section">
                    <h3>Enrolled Users <span class="badge">${users.length}</span></h3>
                    <div class="users-grid">
                        ${users.map(user => `
                            <div class="user-card">
                                <div class="user-image">
                                    <img src="${user.profile_img || defaultAvatarUrl}" alt="Profile Image" onerror="this.src='${defaultAvatarUrl}'">
                                </div>
                                <div class="user-details">
                                    <div class="user-name">${user.firstname} ${user.lastname}</div>
                                    <div class="user-role">${user.role || 'No role specified'}</div>
                                    <div class="user-id">ID: ${user.user_idx}</div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;

        Swal.fire({
            title: 'Logs & Records',
            html: content,
            grow: 'fullscreen',
            showCloseButton: true,
            showConfirmButton: false,
            customClass: {
                container: 'logs-records-modal'
            }
        });
    }

    showGaitEnrollmentPanel() {
        console.log('Showing gait enrollment panel');

        this.clearAccumulatedFeatures();
        document.getElementById('gait-enroll-panel').style.display = 'block';
        document.querySelector('.app-container').classList.add('enrolling');

        if (!this.gaitEnrollEventsSetup) {
            console.log('Setting up gait enrollment specific event listeners...');

            document.getElementById('gait-enroll-close-btn').addEventListener('click', () => {
                console.log('Gait enrollment close button clicked');
                this.hideGaitEnrollmentPanel();
            });

            document.getElementById('gait-enroll-cancel').addEventListener('click', () => {
                console.log('Gait enrollment cancel button clicked');
                this.hideGaitEnrollmentPanel();
            });

            document.getElementById('start-gait-capture').addEventListener('click', () => {
                console.log('Start gait capture button clicked');
                this.startGaitRecording();
            });

            document.getElementById('stop-gait-capture').addEventListener('click', () => {
                console.log('Stop gait capture button clicked');
                this.stopGaitRecording();
            });

            const gaitUploadInput = document.getElementById('gait-enroll-upload-btn');
            if (gaitUploadInput) {
                console.log('Gait upload input found. Setting up change listener.');
                gaitUploadInput.addEventListener('change', (e) => {
                    console.log('Gait upload input change event triggered.');
                    const file = e.target.files[0];
                    if (file) {
                        this.gaitVideoFile = file;
                        console.log('Gait video file selected:', file.name);
                        // Disable recording if a file is selected
                        const startCaptureBtn = document.getElementById('start-gait-capture');
                        const stopCaptureBtn = document.getElementById('stop-gait-capture');
                        if (startCaptureBtn) startCaptureBtn.disabled = true;
                        if (stopCaptureBtn) stopCaptureBtn.disabled = true;
                        console.log('Recording buttons disabled.');
                        this.gaitFrames = [];
                        console.log('Recorded frames cleared.');
                    } else {
                        this.gaitVideoFile = null;
                        console.log('Gait video file selection cancelled');
                        // Re-enable recording if no file is selected
                        const startCaptureBtn = document.getElementById('start-gait-capture');
                        if (startCaptureBtn) startCaptureBtn.disabled = false;
                        console.log('Recording buttons re-enabled.');
                    }
                    const submitBtn = document.getElementById('gait-enroll-submit');
                    if (submitBtn) {
                        submitBtn.disabled = !(this.gaitVideoFile || this.gaitFrames.length > 0);
                        console.log('Submit button disabled state updated:', submitBtn.disabled);
                    } else {
                        console.warn('Gait enrollment submit button not found during upload change!');
                    }
                });
            } else {
                console.warn('Gait upload input element not found!');
            }

            const gaitEnrollSubmitButton = document.getElementById('gait-enroll-submit');
            if (gaitEnrollSubmitButton) {
                console.log('Gait enrollment submit button found. Setting up click listener.');
                gaitEnrollSubmitButton.addEventListener('click', async () => {
                    console.log('"Save Gait" button clicked (from specific listener)');
                    try {
                        const firstname = document.getElementById('gait-enroll-firstname').value.trim();
                        const lastname = document.getElementById('gait-enroll-lastname').value.trim();
                        const role = document.getElementById('gait-enroll-role').value.trim();

                        if (!firstname || !lastname) {
                            this.showLogMessage('Please enter first and last name', 'error');
                            console.log('Validation failed: Missing name');
                            return;
                        }

                        if (this.gaitFrames.length === 0 && !this.gaitVideoFile) {
                            const message = 'Please record gait sequence or upload a video file first';
                            this.showLogMessage(message, 'error');
                            console.log('Validation failed:', message);
                            return;
                        }

                        console.log('Gait data validation passed.');
                        console.log('gaitFrames length:', this.gaitFrames.length);
                        console.log('gaitVideoFile:', this.gaitVideoFile ? this.gaitVideoFile.name : 'None');

                        document.getElementById('gait-enroll-submit').disabled = true;
                        this.showLogMessage('Enrolling gait features...', 'info');
                        console.log('Sending fetch request to /enroll_gait');

                        await this.submitGaitEnrollment();

                    } catch (error) {
                        console.error('Gait enrollment fetch error:', error);
                        this.showLogMessage('Failed to enroll gait features: ' + error.message, 'error');
                    } finally {
                        document.getElementById('gait-enroll-submit').disabled = false;
                        console.log('Fetch request finished, submit button re-enabled.');
                    }
                });
            } else {
                console.warn('Gait enrollment submit button element not found!');
            }

            this.gaitEnrollEventsSetup = true;
            console.log('Gait enrollment specific event listeners set up.');
        }

        this.startGaitEnrollmentCamera().catch(console.error);
    }

    hideGaitEnrollmentPanel() {
        if (this.gaitStream) {
            this.gaitStream.getTracks().forEach(track => track.stop());
            this.gaitStream = null;
        }

        if (this.gaitEnrollCtx) {
            this.gaitEnrollCtx.clearRect(0, 0, this.gaitEnrollCanvas.width, this.gaitEnrollCanvas.height);
        }

        document.getElementById('gait-enroll-panel').style.display = 'none';
        document.querySelector('.app-container').classList.remove('enrolling');

        this.clearAccumulatedFeatures();
        this.resetGaitEnrollmentForm();

        if (!this.currentStream) {
            this.startCamera().catch(console.error);
        }
    }

    async startGaitEnrollmentCamera() {
        try {
            const constraints = {
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                }
            };

            if (this.gaitStream) {
                this.gaitStream.getTracks().forEach(track => track.stop());
            }

            this.gaitStream = await navigator.mediaDevices.getUserMedia(constraints);
            this.gaitEnrollVideo.srcObject = this.gaitStream;

            return new Promise((resolve) => {
                this.gaitEnrollVideo.onloadedmetadata = () => {
                    this.gaitEnrollCanvas.width = this.gaitEnrollVideo.videoWidth;
                    this.gaitEnrollCanvas.height = this.gaitEnrollVideo.videoHeight;
                    this.startGaitPoseDetection();
                    resolve();
                };
            });
        } catch (err) {
            console.error("Gait enrollment camera error:", err);
            this.showLogMessage(`Camera Error: ${err.message}`, "error");
            throw err;
        }
    }

    async startGaitPoseDetection() {
        const processGaitFrame = async () => {
            if (!this.gaitEnrollVideo.srcObject) return;

            if (this.gaitEnrollVideo.readyState === this.gaitEnrollVideo.HAVE_ENOUGH_DATA) {
                this.gaitEnrollCanvas.width = this.gaitEnrollVideo.videoWidth;
                this.gaitEnrollCanvas.height = this.gaitEnrollVideo.videoHeight;
                this.gaitEnrollCtx.drawImage(this.gaitEnrollVideo, 0, 0);

                const imageBlob = await new Promise(resolve => {
                    this.gaitEnrollCanvas.toBlob(resolve, 'image/jpeg', 0.9);
                });

                const formData = new FormData();
                formData.append('image', imageBlob, 'frame.jpg');

                try {
                    const response = await fetch('/pose_detect', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

                    const poseBlob = await response.blob();
                    const imageUrl = URL.createObjectURL(poseBlob);

                    const img = new Image();
                    img.onload = () => {
                        this.gaitEnrollCtx.drawImage(img, 0, 0, this.gaitEnrollCanvas.width, this.gaitEnrollCanvas.height);
                        URL.revokeObjectURL(imageUrl);
                    };
                    img.src = imageUrl;
                } catch (error) {
                    console.error('Pose detection error:', error);
                }
            }

            if (this.gaitEnrollVideo.srcObject) {
                requestAnimationFrame(processGaitFrame);
            }
        };

        processGaitFrame();
    }

    startGaitRecording() {
        if (this.isRecordingGait) return;

        this.isRecordingGait = true;
        this.gaitFrames = [];
        this.recordingStartTime = Date.now();

        document.getElementById('start-gait-capture').style.display = 'none';
        document.getElementById('stop-gait-capture').style.display = 'inline-block';

        const submitBtn = document.getElementById('gait-enroll-submit');
        if (submitBtn) submitBtn.disabled = true;

        // Create a MediaRecorder instance with more compatible settings
        const stream = this.gaitEnrollVideo.srcObject;
        const options = {
            mimeType: 'video/mp4;codecs=h264',
            videoBitsPerSecond: 2500000 // 2.5 Mbps
        };

        // Fallback to webm if mp4 is not supported
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
            options.mimeType = 'video/webm;codecs=h264';
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options.mimeType = 'video/webm';
            }
        }

        this.mediaRecorder = new MediaRecorder(stream, options);

        this.mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                // Create a blob with the appropriate MIME type
                const mimeType = this.mediaRecorder.mimeType;
                this.gaitVideoFile = new Blob([event.data], { type: mimeType });
                console.log('Video file created:', this.gaitVideoFile.size, 'bytes', 'Type:', mimeType);
            }
        };

        this.mediaRecorder.onstop = () => {
            const submitBtn = document.getElementById('gait-enroll-submit');
            if (submitBtn) {
                submitBtn.disabled = false;
                console.log('Submit button enabled after video creation');
            }
        };

        // Start recording
        this.mediaRecorder.start();
        console.log('Started recording gait video with settings:', options);

        // Update progress
        this.recordingInterval = setInterval(() => {
            const progress = Math.min(100, ((Date.now() - this.recordingStartTime) / this.recordingDuration) * 100);
            const progressTextElement = document.querySelector('.progress-text');
            if (progressTextElement) progressTextElement.textContent = `${Math.round(progress)}%`;
            document.querySelector('.progress-fill').style.width = `${progress}%`;

            if (progress >= 100) {
                this.stopGaitRecording();
            }
        }, 100);
    }

    async stopGaitRecording() {
        if (!this.isRecordingGait) return;

        this.isRecordingGait = false;
        clearInterval(this.recordingInterval);
        this.recordingInterval = null;

        document.getElementById('start-gait-capture').style.display = 'inline-block';
        document.getElementById('stop-gait-capture').style.display = 'none';

        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
            console.log('Stopped recording gait video');
        }

        // Reset progress bar
        const progressFill = document.querySelector('.progress-fill');
        const progressText = document.querySelector('.progress-text');
        if (progressFill) progressFill.style.width = '0%';
        if (progressText) progressText.textContent = '0%';
    }

    async submitGaitEnrollment() {
        try {
            const firstname = document.getElementById('gait-enroll-firstname').value.trim();
            const lastname = document.getElementById('gait-enroll-lastname').value.trim();
            const role = document.getElementById('gait-enroll-role').value.trim();

            if (!firstname || !lastname) {
                this.showLogMessage('Please enter first and last name', 'error');
                return;
            }

            if (this.gaitFrames.length === 0 && !this.gaitVideoFile) {
                const message = 'Please record gait sequence or upload a video file first';
                this.showLogMessage(message, 'error');
                return;
            }

            document.getElementById('gait-enroll-submit').disabled = true;
            const submitBtn = document.getElementById('gait-enroll-submit');
            const originalBtnText = submitBtn.innerHTML;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
            this.showLogMessage('Starting gait enrollment process...', 'info');

            const formData = new FormData();
            formData.append('firstname', firstname);
            formData.append('lastname', lastname);
            formData.append('role', role);

            if (this.gaitVideoFile) {
                console.log('Using uploaded video file:', this.gaitVideoFile.name);
                formData.append('video', this.gaitVideoFile);
            } else if (this.gaitFrames.length > 0) {
                console.log('Creating video from recorded frames:', this.gaitFrames.length);
                this.showLogMessage('Creating video from recorded frames...', 'info');
                const videoBlob = await this.createVideoFromFrames();
                const videoFile = new File([videoBlob], 'gait_recording.webm', { type: 'video/webm' });
                formData.append('video', videoFile);
            }

            this.showLogMessage('Saving video and extracting features...', 'info');

            const response = await fetch('/enroll_gait', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            if (!response.ok) {
                throw new Error(result.error || 'Failed to enroll gait features');
            }

            console.log('Gait enrollment result:', result);
            this.showLogMessage(`Successfully enrolled gait for ${firstname} ${lastname}`, 'success');

            this.resetGaitEnrollmentForm();
            this.hideGaitEnrollmentPanel();
            // Refresh user list immediately after successful enrollment
            const users = await this.loadUserList();
            this.renderUserList(users);

        } catch (error) {
            console.error('Gait enrollment error:', error);
            this.showLogMessage(`Gait enrollment failed: ${error.message}`, 'error');
        } finally {
            const submitBtn = document.getElementById('gait-enroll-submit');
            submitBtn.innerHTML = '<i class="fas fa-save"></i> Save Gait';
            submitBtn.disabled = false;
        }
    }

    resetGaitEnrollmentForm() {
        document.getElementById('gait-enroll-firstname').value = '';
        document.getElementById('gait-enroll-lastname').value = '';
        document.getElementById('gait-enroll-role').value = '';


        this.gaitVideoFile = null;
        this.gaitFrames = [];

        const fileInput = document.getElementById('gait-enroll-upload-btn');
        if (fileInput) {
            fileInput.value = '';
        }


        const progressFill = document.querySelector('.progress-fill');
        const progressText = document.querySelector('.progress-text');
        if (progressFill) progressFill.style.width = '0%';
        if (progressText) progressText.textContent = '0%';

        this.isRecordingGait = false;
        if (this.recordingInterval) {
            clearInterval(this.recordingInterval);
            this.recordingInterval = null;
        }


        document.getElementById('start-gait-capture').style.display = 'inline-block';
        document.getElementById('stop-gait-capture').style.display = 'none';
        document.getElementById('gait-enroll-submit').disabled = true;
    }

    async createVideoFromFrames() {
        return new Promise((resolve, reject) => {
            try {
                const canvas = document.createElement('canvas');
                canvas.width = this.gaitEnrollCanvas.width;
                canvas.height = this.gaitEnrollCanvas.height;
                const ctx = canvas.getContext('2d');


                const stream = canvas.captureStream(30);
                const mediaRecorder = new MediaRecorder(stream, {
                    mimeType: 'video/webm;codecs=vp9',
                    videoBitsPerSecond: 2500000 // 2.5 Mbps
                });

                const chunks = [];
                mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
                mediaRecorder.onstop = () => {
                    const blob = new Blob(chunks, { type: 'video/webm' });

                    resolve(blob);
                };

                mediaRecorder.start();

                let frameIndex = 0;
                const frameDuration = 1000 / 30; // 30 FPS
                const drawFrame = () => {
                    if (frameIndex < this.gaitFrames.length) {
                        const img = new Image();
                        img.onload = () => {
                            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                            frameIndex++;
                            setTimeout(drawFrame, frameDuration);
                        };
                        img.src = URL.createObjectURL(this.gaitFrames[frameIndex]);
                    } else {
                        mediaRecorder.stop();
                    }
                };

                drawFrame();
            } catch (error) {
                console.error('Error creating video:', error);
                reject(error);
            }
        });
    }

    async processGaitVideo(file) {
        const video = document.createElement('video');
        video.src = URL.createObjectURL(file);
        video.muted = true; // Mute the video to allow autoplay

        // Create preview container if it doesn't exist
        let previewContainer = document.querySelector('.gait-preview-container');
        if (!previewContainer) {
            previewContainer = document.createElement('div');
            previewContainer.className = 'gait-preview-container';
            const filePreview = document.querySelector('.file-preview');
            filePreview.appendChild(previewContainer);
        }

        // Create canvas for processing preview
        const canvas = document.createElement('canvas');
        canvas.className = 'gait-preview-canvas';
        previewContainer.innerHTML = ''; // Clear previous content
        previewContainer.appendChild(canvas);
        const ctx = canvas.getContext('2d');

        // Create status overlay
        const statusOverlay = document.createElement('div');
        statusOverlay.className = 'gait-status-overlay';
        previewContainer.appendChild(statusOverlay);

        let frameCount = 0;
        let lastFrameTime = performance.now();
        let fpsHistory = [];
        let statusHistory = [];
        let currentStatus = "Processing video...";
        let featureCount = 0;

        const updateStatus = (status, hasFeatures = false) => {
            currentStatus = status;
            if (hasFeatures) featureCount++;

            const now = performance.now();
            statusHistory.push({
                status: currentStatus,
                time: now,
                frame: frameCount
            });
            if (statusHistory.length > 30) statusHistory.shift();

            // Update status display
            statusOverlay.innerHTML = `
                <div class="status-line">${currentStatus}</div>
                <div class="status-line">Feature extractions: ${featureCount}</div>
                <div class="status-line">Frame: ${frameCount}</div>
                <div class="status-line">FPS: ${calculateFPS()}</div>
            `;
        };

        const calculateFPS = () => {
            const now = performance.now();
            const fps = 1000 / (now - lastFrameTime);
            fpsHistory.push(fps);
            if (fpsHistory.length > 30) fpsHistory.shift();
            lastFrameTime = now;
            return (fpsHistory.reduce((a, b) => a + b, 0) / fpsHistory.length).toFixed(1);
        };

        return new Promise((resolve) => {
            video.onloadedmetadata = () => {
                // Set canvas size to match video dimensions
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;

                // Ensure container size matches video aspect ratio
                previewContainer.style.width = '100%';
                previewContainer.style.height = 'auto';
                previewContainer.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;

                // Start playing the video
                video.play().catch(error => {
                    console.error('Error playing video:', error);
                    this.showLogMessage('Error playing video: ' + error.message, 'error');
                });
            };

            video.onended = () => {
                resolve();
            };

            const processFrame = async () => {
                if (video.paused || video.ended) return;

                try {
                    // Draw the current frame
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    frameCount++;

                    // Simulate feature extraction every few frames
                    if (frameCount % 30 === 0) {
                        updateStatus("Extracting features...", true);
                    } else {
                        updateStatus("Processing frame...");
                    }

                    requestAnimationFrame(processFrame);
                } catch (error) {
                    console.error('Error processing frame:', error);
                }
            };

            video.onplay = () => {
                console.log('Video started playing');
                processFrame();
            };

            video.onerror = (error) => {
                console.error('Video error:', error);
                this.showLogMessage('Error loading video: ' + error.message, 'error');
            };
        });
    }

    updateDateTime() {
        const now = new Date();
        const dateString = now.toLocaleDateString();
        const timeString = now.toLocaleTimeString();
        if (this.dateElement && this.timeElement) {
            this.dateElement.textContent = dateString;
            this.timeElement.textContent = timeString;
        } else if (this.datetimeElement) {
            this.datetimeElement.textContent = `${dateString} ${timeString}`;
        }
    }

    clearAccumulatedFeatures() {
        console.log('Clearing accumulated gait features');
        fetch('/clear_gait_features', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        }).catch(error => {
            console.error('Error clearing accumulated features:', error);
        });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new FRS();
}); 