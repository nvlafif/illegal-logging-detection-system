
%% ======================----------------------------------------------------------===================================
%%  CHAINSAW DETECTION SYSTEM v10
%%  + Augmentasi Data Training (6x lipat dataset tanpa rekam ulang)
%%  + Voting Majority Realtime (tekan false positive)
%%  + Integrasi ESP32 + INMP441 via Serial USB
%%  + Output ke ESP32 -> Telegram Bot
%%  
%%  PERUBAHAN v10 vs v9:
%%  [A] AUGMENTASI DATA (fungsi augmentasiSegmen):
%%      Setiap segmen audio diperbanyak menjadi 6 versi:
%%        1. Asli (original)
%%        2. + White noise ringan (~30dB SNR)
%%        3. Time stretch +10% (simulasi RPM lebih cepat)
%%        4. Time stretch -10% (simulasi RPM lebih lambat)
%%        5. Pitch shift +2 semitone
%%        6. Pitch shift -2 semitone
%%      Efek: dataset training 6x lebih besar tanpa rekam ulang
%%  [B] VOTING MAJORITY (di timerCallback):
%%      Alert Telegram hanya dikirim jika >= cfg.voteThr
%%      dari cfg.voteWindow segmen terakhir terdeteksi chainsaw.
%%      Default: 3 dari 5 segmen = chainsaw confirmed.
%%      Status GUI ditambah: "Waspada" (vote belum cukup)
%%      sehingga false positive sesaat tidak memicu alert.
%%  ----------------------------------------------------------
%%  Alur Sistem:
%%  INMP441 -> ESP32 -> USB Serial -> MATLAB -> SVM Klasifikasi
%%                                           -> Voting Majority
%%                                           -> Serial -> ESP32 -> Telegram
%%  ----------------------------------------------------------
%%  Opsi Menu:
%%  1. Training Model (dari file dataset folder) + Augmentasi
%%  2. Uji Realtime (ESP32 + INMP441 via Serial) + Voting
%%  3. Evaluasi Model
%%  4. Visualisasi Feature Space (t-SNE)
%%  ----------------------------------------------------------
%%  KONFIGURASI SERIAL:
%%  - COM Port sesuaikan di cfg.serialPort (misal 'COM3' / '/dev/ttyUSB0')
%%  - Baud Rate: 921600 (sesuai kode ESP32)
%%  - Format paket: header 0xAA 0x55 + 2 byte panjang + data int16 PCM
%%  ----------------------------------------------------------
%%  Referensi Jurnal:
%%  [1] Gnamele et al. (IJACSA 2019) - 13 MFCC + SVM
%%  [2] Mporas et al. (Applied Sciences 2020) - HNR + Spectral
%%  [3] Stefanakis et al. (EUSIPCO 2022) - Augmentasi
%%  [4] Bandara FSC22 (Sensors 2023) - 44100Hz, 5s segment
%% =========================================================

clc; clear; close all;
fprintf('==============================================\n');
fprintf('  CHAINSAW DETECTION SYSTEM v10\n');
fprintf('  + Augmentasi Data + Voting Majority\n');
fprintf('  ESP32 + INMP441 + Telegram Integration\n');
fprintf('==============================================\n\n');
fprintf('1. Training Model (dari dataset folder)\n');
fprintf('2. Uji Realtime (ESP32 + INMP441 via Serial)\n');
fprintf('3. Evaluasi Model\n');
fprintf('4. Visualisasi Pipeline Pemrosesan Audio (14 Tahap)\n');
fprintf('----------------------------------------------\n');
fprintf('Pilih opsi (1/2/3/4): ');
pilihan = input('');

switch pilihan
    case 1
        jalankanTraining();
    case 2
        jalankanRealtimeESP32();
    case 3
        jalankanEvaluasi();
    case 4
        jalankanVisualisasiPipeline();
    otherwise
        fprintf('Pilihan tidak valid.\n');
end


%% =========================================================
%%  FUNGSI UTAMA 1: TRAINING
%% =========================================================
function jalankanTraining()
    cfg = buatKonfigurasi();

    dataFolders = {
        fullfile(cfg.datasetRoot,'chainsaw','dekat'),                    1,'Chainsaw Dekat';
        fullfile(cfg.datasetRoot,'chainsaw','jauh'),                     1,'Chainsaw Jauh';
        fullfile(cfg.datasetRoot,'non_chainsaw_mirip','lawn_mower'),     0,'Lawn Mower';
        fullfile(cfg.datasetRoot,'non_chainsaw_mirip','motor_4tak_low'), 0,'Motor 4Tak Low';
        fullfile(cfg.datasetRoot,'non_chainsaw_mirip','motor_4tak_high'),0,'Motor 4Tak High';
        fullfile(cfg.datasetRoot,'non_chainsaw_mirip','motor_2tak_high'),0, 'Motor 2Tak High';
        fullfile(cfg.datasetRoot,'non_chainsaw_mirip','mesin_listrik'),  0,'Mesin Listrik';
        fullfile(cfg.datasetRoot,'non_chainsaw_mirip','grass_trimmer'),  0,'Grass Trimmer';
        fullfile(cfg.datasetRoot,'non_chainsaw_tidak_mirip'),            0,'Non-CS Tidak Mirip';
    };

    fprintf('\n[1/5] Ekstraksi fitur dataset...\n');
    allFeatures = []; allLabels = []; allKelas = {};

    for i = 1:size(dataFolders,1)
        folderPath = dataFolders{i,1};
        label      = dataFolders{i,2};
        namaKelas  = dataFolders{i,3};

        if ~isfolder(folderPath)
            fprintf('  SKIP (tidak ada): %s\n', namaKelas); continue;
        end
        files = [dir(fullfile(folderPath,'*.wav'));
                 dir(fullfile(folderPath,'*.mp3'));
                 dir(fullfile(folderPath,'*.flac'))];
        if isempty(files)
            fprintf('  SKIP (kosong): %s\n', namaKelas); continue;
        end

        fprintf('  [%s] - %d file...\n', namaKelas, numel(files));
        kelasFitur = [];

        for j = 1:numel(files)
            fp = fullfile(files(j).folder, files(j).name);
            try
                [y, fs] = audioread(fp);
                y = preprocessAudio(y, fs, cfg);
                if isempty(y), continue; end
                fs         = cfg.targetFs;
                segSamples = round(cfg.segLen * fs);
                hopSamples = round((cfg.segLen - cfg.overlap) * fs);
                starts     = 1:hopSamples:(length(y)-segSamples+1);

                for k = 1:numel(starts)
                    seg  = y(starts(k):starts(k)+segSamples-1);
                    feat = ekstrakFitur(seg, fs, cfg);
                    if ~isempty(feat)
                        kelasFitur = [kelasFitur; feat]; %#ok<AGROW>
                    end
                end
            catch
            end
        end

        nSeg        = size(kelasFitur,1);
        allFeatures = [allFeatures; kelasFitur];              %#ok<AGROW>
        allLabels   = [allLabels;   repmat(label,nSeg,1)];    %#ok<AGROW>
        for s = 1:nSeg, allKelas{end+1} = namaKelas; end      %#ok<SAGROW>
        fprintf('    -> %d segmen\n', nSeg);
    end

    fprintf('\n  Total segmen  : %d\n', size(allFeatures,1));
    fprintf('  Dimensi fitur : %d\n',   size(allFeatures,2));
    fprintf('  Chainsaw      : %d\n',   sum(allLabels==1));
    fprintf('  Non-Chainsaw  : %d\n',   sum(allLabels==0));

    fprintf('\n[2/5] Normalisasi fitur (Z-score)...\n');
    mu       = mean(allFeatures, 1);
    sigma    = std(allFeatures,  0, 1);
    sigma(sigma == 0) = 1;
    featNorm = (allFeatures - mu) ./ sigma;

    fprintf('[3/5] Validasi data...\n');
    validIdx  = all(isfinite(featNorm), 2);
    featNorm  = featNorm(validIdx, :);
    allLabels = allLabels(validIdx);
    allKelas  = allKelas(validIdx);
    fprintf('  Segmen valid  : %d\n', sum(validIdx));

    fprintf('\n[4/5] Grid search SVM (harap tunggu)...\n');
    nPos = sum(allLabels == 1); % Chainsaw
    nNeg = sum(allLabels == 0); % Non-Chainsaw

    rasio = nNeg / nPos;  % penalti false negative
    
    costMatrix  = [0, rasio; 1, 0];
    cValues     = [0.1, 1, 10, 100];
    gammaValues = [0.001, 0.01, 0.1, 1];
    bestAcc = 0; bestC = 1; bestGamma = 0.01;

    for ci = 1:numel(cValues)
        for gi = 1:numel(gammaValues)
            try
                svmTmp = fitcsvm(featNorm, allLabels, ...
                    'KernelFunction','rbf', ...
                    'BoxConstraint', cValues(ci), ...
                    'KernelScale',   1/gammaValues(gi), ...
                    'Cost',          costMatrix, ...
                    'Standardize',   false, ...
                    'CrossVal',      'on', ...
                    'KFold',         5);
                acc = 1 - kfoldLoss(svmTmp);
                if acc > bestAcc
                    bestAcc=acc; bestC=cValues(ci); bestGamma=gammaValues(gi);
                end
            catch ME
                fprintf('    [WARN] C=%.1f Gamma=%.4f gagal: %s\n', ...
                        cValues(ci), gammaValues(gi), ME.message);
            end
        end
    end
    fprintf('  Best: C=%.3f, Gamma=%.4f, CV Acc=%.2f%%\n', ...
            bestC, bestGamma, bestAcc*100);

    svmModel = fitcsvm(featNorm, allLabels, ...
        'KernelFunction','rbf', ...
        'BoxConstraint', bestC, ...
        'KernelScale',   1/bestGamma, ...
        'Cost',          costMatrix, ...
        'Standardize',   false);
    svmModelProb = fitPosterior(svmModel);

    fprintf('\n[5/5] Evaluasi training...\n');
    [predLabels, ~] = predict(svmModelProb, featNorm);
    accTrain = sum(predLabels==allLabels)/numel(allLabels)*100;
    fprintf('  Akurasi Training: %.2f%%\n', accTrain);

    cm = confusionmat(allLabels, predLabels);
    TP=cm(2,2); TN=cm(1,1); FP=cm(1,2); FN=cm(2,1);
    precision   = TP/(TP+FP+eps)*100;
    recall      = TP/(TP+FN+eps)*100;
    f1score     = 2*precision*recall/(precision+recall+eps);
    specificity = TN/(TN+FP+eps)*100;

    fprintf('  Precision    : %.2f%%\n', precision);
    fprintf('  Recall       : %.2f%%\n', recall);
    fprintf('  Specificity  : %.2f%%\n', specificity);
    fprintf('  F1-Score     : %.2f%%\n', f1score);
    
    %% ================= VISUALISASI ERROR =================
    fprintf('\n[VISUAL] Misclassified Samples...\n');

    [predLabels, ~] = predict(svmModelProb, featNorm);

    [coeff, score] = pca(featNorm);

    wrongIdx = find(predLabels ~= allLabels);

    figure('Name','Misclassified Data','NumberTitle','off');
    gscatter(score(:,1), score(:,2), allLabels);
    hold on;
    plot(score(wrongIdx,1), score(wrongIdx,2), 'ko', 'MarkerSize',10, 'LineWidth',2);
    title('Data yang Salah Klasifikasi (Lingkaran Hitam)');
    grid on;

    save('chainsaw_model_v10_visual.mat', 'svmModelProb','mu','sigma','cfg', ...
         'bestC','bestGamma','precision','recall','f1score', ...
         'specificity','accTrain','rasio');

    fprintf('\nModel tersimpan: chainsaw_model_v10_visual.mat\n');
    fprintf('Jalankan opsi 2 untuk deteksi realtime.\n');
    fprintf('==============================================\n');
end



%% =========================================================
%%  FUNGSI UTAMA 2: UJI REALTIME DARI ESP32 via SERIAL USB
%%  ----------------------------------------------------------
%%  ARSITEKTUR: Menggunakan MATLAB timer object
%%  Kenapa timer dan bukan while-loop?
%%  - while-loop + drawnow = konflik event GUI yang menyebabkan
%%    figure tertutup sendiri atau "Invalid or deleted object"
%%  - timer object berjalan terpisah dari event loop GUI
%%  - Figure dapat ditutup kapan saja tanpa crash
%%  - Serial dibaca setiap 50ms (callback timerFcn)
%% =========================================================
function jalankanRealtimeESP32()

    if exist('chainsaw_model_v10_visual.mat','file') ~= 2
        error('Model belum ada. Jalankan opsi 1 (Training) dulu.');
    end
    load('chainsaw_model_v10_visual.mat','svmModelProb','mu','sigma','cfg');
    fprintf('\nModel chainsaw_model_v10_visual.mat berhasil dimuat.\n');

    %% ---- KONFIGURASI SERIAL ----
    cfg.serialPort    = 'COM3';   %% Ganti sesuai Device Manager Anda
    cfg.baudRate      = 921600;
    cfg.targetFs      = 16000;
    cfg.segLen        = 2.0;      %% 2 detik per segmen
    cfg.overlap       = 1.0;      %% overlap 1 detik (50%)
    cfg.alertCooldown = 10;       %% detik antar alert Telegram

    %% ---- [v10] KONFIGURASI VOTING MAJORITY ----
    %% Alert hanya dikirim jika minimal voteThr dari voteWindow
    %% segmen terakhir terdeteksi chainsaw.
    %% Naikkan voteThr -> lebih ketat (kurangi FP, mungkin terlambat deteksi)
    %% Turunkan voteThr -> lebih sensitif (lebih cepat, tapi FP bisa naik)
    cfg.voteWindow = 5;   %% pertimbangkan 5 segmen terakhir
    cfg.voteThr    = 3;   %% minimal 3 dari 5 = chainsaw confirmed

    segSamples = round(cfg.segLen  * cfg.targetFs);   %% = 32000
    hopSamples = round(cfg.overlap * cfg.targetFs);   %% = 16000

    %% ---- BUKA KONEKSI SERIAL ----
    fprintf('\nMembuka Serial %s @ %d baud...\n', cfg.serialPort, cfg.baudRate);

    existingPorts = instrfind('Type','serial','Port',cfg.serialPort);
    if ~isempty(existingPorts)
        fclose(existingPorts);
        delete(existingPorts);
    end

    try
        s = serial(cfg.serialPort, ...
            'BaudRate',        cfg.baudRate, ...
            'InputBufferSize', 262144, ...
            'Timeout',         1);
        fopen(s);
        fprintf('Serial terhubung.\n');
    catch ME
        fprintf('GAGAL buka serial: %s\n', ME.message);
        fprintf('Pastikan COM port benar dan ESP32 terhubung.\n');
        return;
    end

    %% ---- BUAT FIGURE ----
    fig = figure('Name','Chainsaw Detector v5 - Realtime', ...
                 'NumberTitle','off', ...
                 'Position',[50 50 1100 650], ...
                 'Resize','off');

    %% Panel waveform
    ax1 = subplot(3,2,[1 2]);
    waveInit = zeros(1, segSamples);
    hWave    = plot(ax1, 1:segSamples, waveInit, 'Color',[0.2 0.4 0.8], 'LineWidth',0.5);
    title(ax1,'Waveform Audio Realtime (INMP441)');
    xlabel(ax1,'Sampel'); ylabel(ax1,'Amplitudo');
    ylim(ax1,[-1.1 1.1]);
    xlim(ax1,[1 segSamples]);
    grid(ax1,'on');

    %% Panel bar probabilitas
    ax2 = subplot(3,2,3);
    hBar = bar(ax2, [0 0], 'FaceColor',[0.2 0.6 1]);
    set(ax2,'XTickLabel',{'Non-Chainsaw','Chainsaw'});
    ylabel(ax2,'Probabilitas (%)');
    title(ax2,'Hasil Klasifikasi SVM');
    ylim(ax2,[0 110]);
    grid(ax2,'on');

    %% Panel history probabilitas
    ax3 = subplot(3,2,4);
    nHistory = 30;
    histProb = zeros(1, nHistory);
    hHist    = plot(ax3, 1:nHistory, histProb, 'b-o', ...
                    'MarkerSize',4, 'LineWidth',1.5);
    hold(ax3,'on');
    hThresh  = plot(ax3, [1 nHistory],[50 50],'--r','LineWidth',1.5);
    hold(ax3,'off');
    title(ax3,'Riwayat P(Chainsaw) %');
    xlabel(ax3,'Segmen ke-'); ylabel(ax3,'%');
    ylim(ax3,[0 105]); grid(ax3,'on');

    %% Panel status
    ax4 = subplot(3,2,[5 6]);
    axis(ax4,'off');
    set(ax4,'Color',[0.95 0.95 0.95]);
    hStatus = text(ax4, 0.5, 0.6, 'MENUNGGU DATA...', ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','middle', ...
        'FontSize',20, 'FontWeight','bold', ...
        'Color',[0.5 0.5 0.5], ...
        'Units','normalized');
    hSegInfo = text(ax4, 0.5, 0.2, 'Segmen: 0 | Buffer: 0/32000', ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','middle', ...
        'FontSize',10, 'Color',[0.4 0.4 0.4], ...
        'Units','normalized');
    title(ax4,'Status Deteksi');

    %% ---- SHARED STATE via appdata ----
    setappdata(fig, 'audioBuffer',    zeros(0,1));
    setappdata(fig, 'segCount',       0);
    setappdata(fig, 'histProb',       zeros(1,nHistory));
    alertClockInit = clock();
    alertClockInit(6) = alertClockInit(6) - (cfg.alertCooldown + 1);
    setappdata(fig, 'lastAlertClock', alertClockInit);
    setappdata(fig, 'serial',         s);
    setappdata(fig, 'running',        true);
    setappdata(fig, 'HEADER',         [hex2dec('AA'), hex2dec('55')]);
    %% [v10] Buffer voting: menyimpan voteWindow prediksi terakhir (0/1)
    setappdata(fig, 'voteBuffer',     zeros(1, cfg.voteWindow));

    %% ---- KIRIM START KE ESP32 ----
    fprintf('Menunggu ESP32 READY (max 8 detik)...\n');
    t0 = clock();
    espReady = false;
    while etime(clock(),t0) < 8
        pause(0.2);
        if s.BytesAvailable > 0
            raw = fread(s, s.BytesAvailable, 'uint8');
            if ~isempty(strfind(char(raw(:)'), 'READY'))
                espReady = true;
                break;
            end
        end
    end
    if espReady
        fprintf('ESP32 READY diterima.\n');
    else
        fprintf('[INFO] READY tidak diterima, lanjut tetap.\n');
    end

    %% Flush sisa byte sebelum START
    if s.BytesAvailable > 0
        fread(s, s.BytesAvailable, 'uint8');
    end

    fprintf(s, 'START\n');
    pause(0.3);
    fprintf('START dikirim ke ESP32.\n');
    fprintf('\n==============================================\n');
    fprintf('  DETEKSI REALTIME BERJALAN\n');
    fprintf('  Tutup jendela figure untuk berhenti\n');
    fprintf('==============================================\n');

    %% ---- BUAT TIMER OBJECT ----
    %% Timer dipanggil setiap 50ms untuk baca serial dan update GUI
    %% Ini JAUH lebih aman dari while-loop karena:
    %% 1. Tidak memblok event loop MATLAB
    %% 2. Figure bisa ditutup kapan saja
    %% 3. Tidak ada konflik drawnow vs CloseRequestFcn
    t = timer(...
        'ExecutionMode', 'fixedRate', ...
        'Period',        0.05, ...
        'BusyMode',      'drop', ...
        'TimerFcn',      @(~,~) timerCallback(fig, ax1, ax2, ax3, ax4, ...
                                              hWave, hBar, hHist, hStatus, hSegInfo, ...
                                              svmModelProb, mu, sigma, cfg, ...
                                              segSamples, hopSamples, nHistory), ...
        'ErrorFcn',      @(obj,~) timerErrorRestart(obj), ...
        'StopFcn',       @(~,~) timerStop());

    %% Simpan referensi timer ke appdata agar bisa dihentikan dari CloseRequestFcn
    setappdata(fig, 'timer', t);

    %% CloseRequestFcn - hentikan timer lalu tutup figure
    set(fig, 'CloseRequestFcn', @(src,~) tutupFigureTimer(src));

    %% Jalankan timer
    start(t);

    %% Tunggu sampai figure ditutup user
    %% uiwait adalah cara BENAR untuk tunggu figure - tidak ada while-loop!
    try
        uiwait(fig);
    catch
    end

    %% ---- CLEANUP SETELAH FIGURE DITUTUP ----
    fprintf('\nMenghentikan sistem...\n');

    %% Hentikan dan hapus timer
    try
        if isvalid(t)
            stop(t);
            delete(t);
        end
    catch
    end

    %% Tutup serial
    try
        if isvalid(s) && strcmp(s.Status,'open')
            fprintf(s, 'STOP\n');
            pause(0.2);
            fclose(s);
        end
        delete(s);
        fprintf('Koneksi serial ditutup.\n');
    catch
    end

    segCount = 0;
    try
        if ishghandle(fig)
            segCount = getappdata(fig, 'segCount');
            delete(fig);
        end
    catch
    end
    fprintf('Selesai. Total segmen: %d\n', segCount);
end


%% =========================================================
%%  TIMER CALLBACK - Dipanggil setiap 50ms
%%  Baca serial -> akumulasi buffer -> inferensi SVM -> update GUI
%% =========================================================
function timerCallback(fig, ax1, ax2, ax3, ax4, ...
                       hWave, hBar, hHist, hStatus, hSegInfo, ...
                       svmModelProb, mu, sigma, cfg, ...
                       segSamples, hopSamples, nHistory)

    %% Guard: keluar jika figure sudah tidak ada
    if ~ishghandle(fig), return; end
    if ~getappdata(fig,'running'), return; end

    %% Ambil state dari appdata
    s              = getappdata(fig, 'serial');
    audioBuffer    = getappdata(fig, 'audioBuffer');
    segCount       = getappdata(fig, 'segCount');
    histProb       = getappdata(fig, 'histProb');
    lastAlertClock = getappdata(fig, 'lastAlertClock');
    HEADER         = getappdata(fig, 'HEADER');
    voteBuffer     = getappdata(fig, 'voteBuffer');  %% [v10] voting buffer

    %% ---- BACA DATA DARI SERIAL ----
    try
        nAv = s.BytesAvailable;
    catch
        return;
    end

    if nAv >= 2
        try
            %% Bulatkan ke kelipatan 2 (int16 = 2 bytes)
            nBaca    = nAv - mod(nAv,2);
            rawBytes = fread(s, nBaca, 'uint8');
            rawBytes = uint8(rawBytes(:));

            if ~isempty(rawBytes) && length(rawBytes) >= 2
                newSamples = bacaSampel(rawBytes, HEADER);
                if ~isempty(newSamples)
                    audioBuffer = [audioBuffer; newSamples];
                end
            end
        catch
            %% Serial error - skip iterasi ini
            return;
        end
    end

    %% ---- UPDATE WAVEFORM ----
    if ~ishghandle(fig), return; end
    try
        bufLen = length(audioBuffer);
        if bufLen >= segSamples
            waveData = audioBuffer(end-segSamples+1:end);
        elseif bufLen > 0
            waveData = [zeros(segSamples-bufLen, 1); audioBuffer];
        else
            waveData = zeros(segSamples, 1);
        end
        set(hWave, 'YData', double(waveData(:))');

        %% Update info buffer
        pct = min(100, round(bufLen/segSamples*100));
        set(hSegInfo, 'String', ...
            sprintf('Segmen: %d | Buffer: %d/%d (%.0f%%)', ...
                    segCount, bufLen, segSamples, pct));
    catch
        return;
    end

    %% ---- INFERENSI SVM ----
    while length(audioBuffer) >= segSamples

        if ~ishghandle(fig), break; end

        seg         = audioBuffer(1:segSamples);
        audioBuffer = audioBuffer(hopSamples+1:end);

        maxV = max(abs(seg));
        if maxV < 1e-6
            continue;
        end
        segN = seg / maxV;

        %% Ekstraksi fitur
        feat = ekstrakFitur(segN, cfg.targetFs, cfg);
        if isempty(feat), continue; end

        %% Klasifikasi SVM
        try
            featNorm = (feat - mu) ./ sigma;
            [predLabel, scores] = predict(svmModelProb, featNorm);
            probCS    = scores(2) * 100;
            probNonCS = scores(1) * 100;
        catch
            continue;
        end

        segCount = segCount + 1;
        histProb = [histProb(2:end), probCS];

        %% ---- [v10] VOTING MAJORITY ----
        %% Geser buffer, masukkan prediksi segmen ini (0 atau 1)
        voteBuffer   = [voteBuffer(2:end), predLabel];
        nVoteCS      = sum(voteBuffer == 1);
        voteConfirmed = (nVoteCS >= cfg.voteThr);

        if ~ishghandle(fig), break; end

        %% Update bar probabilitas
        try
            set(hBar, 'YData', [probNonCS, probCS]);
            if voteConfirmed
                set(hBar, 'FaceColor', [0.85 0.15 0.15]);   %% merah = confirmed
            elseif predLabel == 1
                set(hBar, 'FaceColor', [0.95 0.60 0.10]);   %% oranye = waspada
            else
                set(hBar, 'FaceColor', [0.15 0.65 0.15]);   %% hijau = aman
            end
        catch, end

        %% Update history
        try
            set(hHist, 'YData', histProb);
        catch, end

        %% Update status teks + kirim alert
        try
            if voteConfirmed
                %% ==========================================
                %% STATUS MERAH: CHAINSAW CONFIRMED oleh vote
                %% ==========================================
                set(hStatus, ...
                    'String',   sprintf('!! CHAINSAW TERDETEKSI !!  %.1f%%  [Vote %d/%d]', ...
                                        probCS, nVoteCS, cfg.voteWindow), ...
                    'Color',    [0.85 0.05 0.05], ...
                    'FontSize', 20);
                set(ax4, 'Color', [1.0 0.88 0.88]);

                %% Kirim alert ke ESP32 (dengan cooldown)
                sekarang     = clock();
                selisihDetik = etime(sekarang, lastAlertClock);
                if selisihDetik >= cfg.alertCooldown
                    try
                        alertMsg = sprintf('ALERT:%.1f\n', probCS);
                        fwrite(s, uint8(alertMsg), 'uint8');
                        lastAlertClock = sekarang;
                        fprintf('[%s] CHAINSAW CONFIRMED P=%.1f%% Vote=%d/%d -> Alert ESP32\n', ...
                            datestr(now,'HH:MM:SS'), probCS, nVoteCS, cfg.voteWindow);
                    catch alertErr
                        fprintf('[WARN] Gagal kirim alert: %s\n', alertErr.message);
                    end
                else
                    fprintf('[%s] CHAINSAW CONFIRMED P=%.1f%% Vote=%d/%d (cooldown %.0f/%ds)\n', ...
                        datestr(now,'HH:MM:SS'), probCS, nVoteCS, cfg.voteWindow, ...
                        selisihDetik, cfg.alertCooldown);
                end

            elseif predLabel == 1
                %% ==========================================
                %% STATUS ORANYE: Terdeteksi tapi vote belum cukup
                %% Ini yang menekan false positive sesaat!
                %% ==========================================
                set(hStatus, ...
                    'String',   sprintf('~ Waspada: Terdeteksi %.1f%%  [Vote %d/%d]', ...
                                        probCS, nVoteCS, cfg.voteWindow), ...
                    'Color',    [0.85 0.50 0.00], ...
                    'FontSize', 18);
                set(ax4, 'Color', [1.0 0.97 0.88]);
                fprintf('[Seg %d] P(CS)=%.1f%% -> WASPADA (vote %d/%d, belum kirim alert)\n', ...
                    segCount, probCS, nVoteCS, cfg.voteWindow);

            else
                %% ==========================================
                %% STATUS HIJAU: Aman
                %% ==========================================
                set(hStatus, ...
                    'String',   sprintf('Aman  -  Bukan Chainsaw  (%.1f%%)', probCS), ...
                    'Color',    [0.05 0.55 0.05], ...
                    'FontSize', 18);
                set(ax4, 'Color', [0.90 1.00 0.90]);
                fprintf('[Seg %d] P(CS)=%.1f%% -> Aman\n', segCount, probCS);
            end
        catch, end

    end %% end while inferensi

    %% Simpan state kembali ke appdata
    if ishghandle(fig)
        setappdata(fig, 'audioBuffer',    audioBuffer);
        setappdata(fig, 'segCount',       segCount);
        setappdata(fig, 'histProb',       histProb);
        setappdata(fig, 'lastAlertClock', lastAlertClock);
        setappdata(fig, 'voteBuffer',     voteBuffer);  %% [v10]
    end
end


%% =========================================================
%%  FUNGSI: PARSE BYTES DARI SERIAL -> SAMPEL FLOAT
%%  Support: header 0xAA55 (firmware lama) + raw PCM (firmware baru)
%% =========================================================
function samples = bacaSampel(rawBytes, HEADER)
    samples = [];
    if length(rawBytes) < 2, return; end

    %% Coba cari header 0xAA 0x55
    headerFound = false;
    for i = 1:(length(rawBytes)-3)
        if rawBytes(i)==uint8(HEADER(1)) && rawBytes(i+1)==uint8(HEADER(2))
            lenH    = double(rawBytes(i+2));
            lenL    = double(rawBytes(i+3));
            dataLen = lenH*256 + lenL;
            endIdx  = i + 3 + dataLen;
            if dataLen >= 2 && dataLen <= 4096 && endIdx <= length(rawBytes)
                dataBytes = rawBytes(i+4:endIdx);
                if mod(length(dataBytes),2) == 0
                    pcm     = typecast(dataBytes, 'int16');
                    samples = double(pcm(:)) / 32768.0;
                    headerFound = true;
                end
            end
            break;
        end
    end

    %% Fallback: baca raw sebagai int16 PCM
    if ~headerFound
        n = length(rawBytes) - mod(length(rawBytes),2);
        if n >= 2
            pcm     = typecast(rawBytes(1:n), 'int16');
            samples = double(pcm(:)) / 32768.0;
        end
    end

    %% Buang segmen yang semua nol
    if ~isempty(samples) && max(abs(samples)) < 1e-9
        samples = [];
    end
end


%% =========================================================
%%  CALLBACK: Tombol X diklik user
%% =========================================================
function tutupFigureTimer(fig)
    try
        setappdata(fig, 'running', false);
        t = getappdata(fig, 'timer');
        if ~isempty(t) && isvalid(t)
            stop(t);
        end
    catch
    end
    %% uiresume membangunkan uiwait di jalankanRealtimeESP32
    try
        uiresume(fig);
    catch
    end
end


%% =========================================================
%%  TIMER STOP CALLBACK - Sengaja kosong
%%  uiresume HANYA boleh dipanggil dari tutupFigureTimer
%% =========================================================
function timerStop()
    %% Tidak melakukan apa-apa
    %% Ini mencegah sistem berhenti saat timer stop karena sebab internal
end


%% =========================================================
%%  TIMER ERROR CALLBACK - Restart timer jika ada error
%%  Tanpa ini, satu error kecil di timerCallback akan
%%  menghentikan timer selamanya dan sistem berhenti
%% =========================================================
function timerErrorRestart(t)
    try
        if isvalid(t) && strcmp(t.Running, 'off')
            start(t);   %% Restart timer otomatis
        end
    catch
    end
end


%% =========================================================
%%  FUNGSI UTAMA 3: EVALUASI MODEL
%% =========================================================
function jalankanEvaluasi()
    if exist('chainsaw_model_v10_visual.mat','file') ~= 2
        error('Model belum ada. Jalankan opsi 1 dulu.');
    end
    load('chainsaw_model_v10_visual.mat','svmModelProb','mu','sigma','cfg');

    cfg.datasetRoot = 'Data Uji';
    
    dataFolders = {
        fullfile(cfg.datasetRoot,'chainsaw','dekat'),                    1,'Chainsaw Dekat';
        fullfile(cfg.datasetRoot,'chainsaw','jauh'),                     1,'Chainsaw Jauh';
        fullfile(cfg.datasetRoot,'non_chainsaw_mirip','lawn_mower'),     0,'Lawn Mower';
        fullfile(cfg.datasetRoot,'non_chainsaw_mirip','motor_4_tak_low'), 0,'Motor 4Tak Low';
        fullfile(cfg.datasetRoot,'non_chainsaw_mirip','motor_4_tak_high'),0,'Motor 4Tak High';
        fullfile(cfg.datasetRoot,'non_chainsaw_mirip','motor_2_tak_high'),0,'Motor 2Tak High';
        fullfile(cfg.datasetRoot,'non_chainsaw_mirip','mesin_listrik'),  0,'Mesin Listrik';
        fullfile(cfg.datasetRoot,'non_chainsaw_mirip','grass_trimmer'),  0,'Grass Trimmer';
        fullfile(cfg.datasetRoot,'non_chainsaw_tidak_mirip'),            0,'Non-CS Tidak Mirip';
    };

    fprintf('\nEkstraksi fitur untuk evaluasi...\n');
    allFeatures = []; allLabels = []; allKelas = {};

    for i = 1:size(dataFolders,1)
        folderPath = dataFolders{i,1};
        label      = dataFolders{i,2};
        namaKelas  = dataFolders{i,3};
        if ~isfolder(folderPath), continue; end
        files = [dir(fullfile(folderPath,'*.wav'));
                 dir(fullfile(folderPath,'*.mp3'))];
        if isempty(files), continue; end
        for j = 1:numel(files)
            fp = fullfile(files(j).folder,files(j).name);
            try
                [y,fs] = audioread(fp);
                y = preprocessAudio(y,fs,cfg);
                if isempty(y), continue; end
                fs = cfg.targetFs;
                segSamples = round(cfg.segLen*fs);
                hopSamples = round((cfg.segLen-cfg.overlap)*fs);
                starts = 1:hopSamples:(length(y)-segSamples+1);
                for k = 1:numel(starts)
                    seg  = y(starts(k):starts(k)+segSamples-1);
                    feat = ekstrakFitur(seg,fs,cfg);
                    if ~isempty(feat)
                        allFeatures     = [allFeatures; feat];   %#ok<AGROW>
                        allLabels       = [allLabels; label];    %#ok<AGROW>
                        allKelas{end+1} = namaKelas;             %#ok<SAGROW>
                    end
                end
            catch
            end
        end
        fprintf('  OK: %s\n', namaKelas);
    end

    featNorm = (allFeatures - mu) ./ sigma;
    [predLabels, scores] = predict(svmModelProb, featNorm);
    probCS = scores(:,2);

    uniqueKelas = unique(allKelas,'stable');
    nKelas      = numel(uniqueKelas);

    figure('Name','Evaluasi v7','NumberTitle','off','Position',[40 40 1200 820]);

    ax1 = subplot(2,3,1);
    cm  = confusionmat(allLabels, predLabels);
    imagesc(cm);
    colormap(ax1,[0.93 0.93 0.93; 0.18 0.52 0.88]);
    for r=1:2, for c=1:2
        text(c,r,num2str(cm(r,c)),'HorizontalAlignment','center', ...
            'FontSize',16,'FontWeight','bold');
    end; end
    set(ax1,'XTick',[1 2],'XTickLabel',{'Non-CS','Chainsaw'}, ...
            'YTick',[1 2],'YTickLabel',{'Non-CS','Chainsaw'});
    xlabel('Prediksi'); ylabel('Ground Truth');
    TP=cm(2,2); TN=cm(1,1); FP=cm(1,2); FN=cm(2,1);
    acc=(TP+TN)/sum(cm(:))*100;
    title(sprintf('Confusion Matrix - Akurasi: %.1f%%',acc));

    subplot(2,3,2);
    [X,Y,~,AUC] = perfcurve(allLabels, probCS, 1);
    plot(X,Y,'b-','LineWidth',2); hold on;
    plot([0 1],[0 1],'k--'); xlabel('FPR'); ylabel('TPR');
    title(sprintf('ROC Curve  AUC = %.4f',AUC)); grid on; axis square;

    subplot(2,3,3); hold on;
    legStr = {};
    for ci=1:nKelas
        idx = strcmp(allKelas,uniqueKelas{ci});
        if sum(idx)<2, continue; end
        [f,xi] = ksdensity(probCS(idx),'BoundaryCorrection','reflection');
        plot(xi,f,'LineWidth',1.5);
        legStr{end+1} = uniqueKelas{ci}; %#ok<SAGROW>
    end
    plot([0.5 0.5],[0 10],'--k','LineWidth',1.5);
    xlabel('P(Chainsaw)'); ylabel('Densitas');
    title('Distribusi P(Chainsaw) per Kelas');
    legend(legStr,'FontSize',7,'Location','best'); grid on;

    ax4 = subplot(2,3,4);
    classAcc = zeros(1,nKelas);
    for ci=1:nKelas
        idx = strcmp(allKelas,uniqueKelas{ci});
        classAcc(ci) = sum(predLabels(idx)==allLabels(idx))/sum(idx)*100;
    end
    b = bar(classAcc,'FaceColor','flat');
    for ci=1:nKelas
        if classAcc(ci)>=90, b.CData(ci,:)=[0.2 0.75 0.3];
        elseif classAcc(ci)>=70, b.CData(ci,:)=[0.9 0.75 0.1];
        else, b.CData(ci,:)=[0.85 0.2 0.2]; end
    end
    set(ax4,'XTick',1:nKelas,'XTickLabel',uniqueKelas,'XTickLabelRotation',30);
    ylabel('Akurasi (%)'); ylim([0 108]);
    title('Akurasi per Kelas'); grid on;
    line([0.5 nKelas+0.5],[90 90],'Color','r','LineStyle','--','LineWidth',1.2);
    for ci=1:nKelas
        text(ci,classAcc(ci)+1.5,sprintf('%.0f%%',classAcc(ci)), ...
            'HorizontalAlignment','center','FontSize',8,'FontWeight','bold');
    end

    ax5 = subplot(2,3,5); hold on;
    for ci=1:nKelas
        idx  = strcmp(allKelas,uniqueKelas{ci});
        gambarBoxPlot(ax5, probCS(idx)*100, ci);
    end
    plot(ax5,[0.5 nKelas+0.5],[50 50],'--k','LineWidth',1.5);
    set(ax5,'XTick',1:nKelas,'XTickLabel',uniqueKelas,'XTickLabelRotation',30);
    ylabel('P(Chainsaw) %'); ylim([-5 108]);
    title('Box Plot P(Chainsaw) per Kelas'); grid on;

    subplot(2,3,6); axis off;
    precision   = TP/(TP+FP+eps)*100;
    recall      = TP/(TP+FN+eps)*100;
    f1          = 2*precision*recall/(precision+recall+eps);
    specificity = TN/(TN+FP+eps)*100;
    metrics = {'Akurasi','Precision','Recall','Specificity','F1-Score','AUC'};
    values  = [acc, precision, recall, specificity, f1, AUC*100];
    yPos    = linspace(0.85,0.15,numel(metrics));
    for m=1:numel(metrics)
        if values(m)>=90, clr=[0.0 0.55 0.0];
        elseif values(m)>=70, clr=[0.65 0.45 0.0];
        else, clr=[0.75 0.0 0.0]; end
        text(0.05,yPos(m),metrics{m},'Units','normalized','FontSize',10,'Color',[0.3 0.3 0.3]);
        text(0.72,yPos(m),sprintf('%.2f%%',values(m)),'Units','normalized', ...
            'FontSize',11,'FontWeight','bold','Color',clr,'HorizontalAlignment','right');
    end
    title('Ringkasan Metrik','FontSize',11);
    annotation('textbox',[0.3 0.97 0.4 0.03],'String','Evaluasi Chainsaw Detection v7','FontSize',13,'FontWeight','bold','EdgeColor','none','HorizontalAlignment','center');

    fprintf('\n==============================================\n');
    fprintf('  RINGKASAN EVALUASI v7\n');
    fprintf('==============================================\n');
    fprintf('  Akurasi     : %.2f%%\n', acc);
    fprintf('  Precision   : %.2f%%\n', precision);
    fprintf('  Recall      : %.2f%%\n', recall);
    fprintf('  Specificity : %.2f%%\n', specificity);
    fprintf('  F1-Score    : %.2f%%\n', f1);
    fprintf('  AUC         : %.4f\n',   AUC);
    for ci=1:nKelas
        fprintf('  %-25s : %.1f%%\n', uniqueKelas{ci}, classAcc(ci));
    end
    fprintf('==============================================\n');
end

function jalankanVisualisasiPipeline()
%% =========================================================
%%  FUNGSI UTAMA 4: VISUALISASI PIPELINE PEMROSESAN AUDIO
%%  14 Tahap lengkap dari raw audio hingga hasil training SVM
%% =========================================================

    cfg = buatKonfigurasi();

    %% ---- Cari 1 file audio chainsaw/dekat ----
    folderChainsaw = fullfile(cfg.datasetRoot, 'chainsaw', 'dekat');
    if ~isfolder(folderChainsaw)
        fprintf('[ERROR] Folder tidak ditemukan: %s\n', folderChainsaw);
        fprintf('Pastikan folder dataset/chainsaw/dekat ada dan berisi file audio.\n');
        return;
    end

    files = [dir(fullfile(folderChainsaw,'*.wav')); ...
             dir(fullfile(folderChainsaw,'*.mp3')); ...
             dir(fullfile(folderChainsaw,'*.flac'))];

    if isempty(files)
        fprintf('[ERROR] Tidak ada file audio di: %s\n', folderChainsaw);
        return;
    end

    filePath = fullfile(files(1).folder, files(1).name);
    fprintf('\n==============================================\n');
    fprintf('  VISUALISASI PIPELINE PEMROSESAN AUDIO\n');
    fprintf('  File: %s\n', files(1).name);
    fprintf('==============================================\n');

    %% ============================================================
    %%  TAHAP 0: Baca file audio
    %% ============================================================
    [yRaw, fsOri] = audioread(filePath);

    %% ============================================================
    %%  TAHAP 1: Sinyal Audio Mentah
    %% ============================================================
    fprintf('[1/14] Sinyal audio mentah...\n');
    if size(yRaw,2) > 1, yMono = mean(yRaw,2); else, yMono = yRaw; end
    tRaw = (0:length(yMono)-1) / fsOri;

    fig1 = figure('Name','Tahap 1 - Sinyal Audio Mentah','NumberTitle','off', ...
                  'Position',[50 600 1200 300]);
    plot(tRaw, yMono, 'Color',[0.1 0.4 0.8], 'LineWidth',0.5);
    xlabel('Waktu (detik)'); ylabel('Amplitudo');
    title(sprintf('TAHAP 1: Sinyal Audio Mentah  |  File: %s  |  Fs ori = %d Hz  |  Durasi = %.2f detik', ...
          files(1).name, fsOri, length(yMono)/fsOri), 'Interpreter','none');
    grid on; xlim([tRaw(1) tRaw(end)]);
    annotation(fig1,'textbox',[0 0 1 0.06],'String', ...
        sprintf('Kanal: %d | Sampel: %d | Max Amp: %.4f | Min Amp: %.4f', ...
        size(yRaw,2), length(yMono), max(yMono), min(yMono)), ...
        'EdgeColor','none','HorizontalAlignment','center','FontSize',8);

    %% ============================================================
    %%  TAHAP 2: Sinyal Setelah Pre-processing
    %% ============================================================
    fprintf('[2/14] Pre-processing (mono, resample, normalisasi)...\n');
    if fsOri ~= cfg.targetFs
        yPrep = resample(yMono, cfg.targetFs, fsOri);
    else
        yPrep = yMono;
    end
    maxVal = max(abs(yPrep));
    if maxVal > 1e-8, yPrep = yPrep / maxVal; end
    fs = cfg.targetFs;
    tPrep = (0:length(yPrep)-1) / fs;

    fig2 = figure('Name','Tahap 2 - Setelah Pre-processing','NumberTitle','off', ...
                  'Position',[50 600 1200 300]);
    subplot(1,2,1);
    plot(tRaw, yMono, 'Color',[0.7 0.7 0.7],'LineWidth',0.5); hold on;
    plot(tPrep, yPrep, 'Color',[0.1 0.4 0.8],'LineWidth',0.7);
    legend('Sebelum','Sesudah'); xlabel('Waktu (s)'); ylabel('Amplitudo');
    title('Perbandingan Sebelum vs Sesudah Pre-proc'); grid on;
    subplot(1,2,2);
    plot(tPrep, yPrep, 'Color',[0.1 0.4 0.8],'LineWidth',0.7);
    xlabel('Waktu (detik)'); ylabel('Amplitudo (Ternormalisasi)');
    title(sprintf('TAHAP 2: Setelah Pre-processing  |  Fs = %d Hz  |  Sampel = %d', ...
          fs, length(yPrep)));
    grid on; xlim([tPrep(1) tPrep(end)]);
    sgtitle('TAHAP 2: Pre-processing (Mono → Resample → Normalisasi)', ...
            'FontSize',12,'FontWeight','bold');

    %% ============================================================
    %%  TAHAP 3: Segmentasi dan Overlap
    %% ============================================================
    fprintf('[3/14] Segmentasi dan overlap...\n');
    segSamples = round(cfg.segLen * fs);
    hopSamples = round((cfg.segLen - cfg.overlap) * fs);

    if length(yPrep) < segSamples
        yPrep = [yPrep; zeros(segSamples - length(yPrep), 1)];
    end

    starts = 1:hopSamples:(length(yPrep) - segSamples + 1);
    nSeg   = numel(starts);

    fig3 = figure('Name','Tahap 3 - Segmentasi dan Overlap','NumberTitle','off', ...
                  'Position',[50 600 1200 450]);
    plot(tPrep, yPrep, 'Color',[0.1 0.4 0.8],'LineWidth',0.5); hold on;
    colors3 = lines(min(nSeg,6));
    for k = 1:min(nSeg,6)
        s = starts(k); e = s + segSamples - 1;
        if e > length(yPrep), e = length(yPrep); end
        xS = (s-1)/fs; xE = (e-1)/fs;
        patch([xS xE xE xS],[1.1 1.1 -1.1 -1.1],colors3(k,:), ...
              'FaceAlpha',0.12,'EdgeColor',colors3(k,:),'LineWidth',1.5);
        text((xS+xE)/2, 1.0, sprintf('Seg %d',k),'HorizontalAlignment','center', ...
             'FontSize',8,'Color',colors3(k,:),'FontWeight','bold');
    end
    ylim([-1.2 1.25]);
    xlabel('Waktu (detik)'); ylabel('Amplitudo');
    title(sprintf('TAHAP 3: Segmentasi + Overlap  |  Panjang Seg = %.1fs  |  Overlap = %.1fs  |  Total Segmen = %d', ...
          cfg.segLen, cfg.overlap, nSeg));
    grid on; hold off;
    annotation(fig3,'textbox',[0 0 1 0.06],'String', ...
        sprintf('Segmen: %d sampel (%.1f detik) | Hop: %d sampel (%.1f detik) | Total segmen: %d', ...
        segSamples, cfg.segLen, hopSamples, cfg.overlap, nSeg), ...
        'EdgeColor','none','HorizontalAlignment','center','FontSize',9);

    %% Pilih segmen pertama untuk analisis selanjutnya
    seg = yPrep(starts(1):starts(1)+segSamples-1);
    maxSeg = max(abs(seg)); if maxSeg > 1e-8, seg = seg / maxSeg; end
    tSeg = (0:segSamples-1)/fs;

    %% ============================================================
    %%  TAHAP 4: Framing dan Windowing
    %% ============================================================
    fprintf('[4/14] Framing dan windowing...\n');
    
    %%===FRAMING=====
    frameLen = round(cfg.frameLen * fs);
    frameHop = round(cfg.frameHop * fs);

    frames   = buffer(seg, frameLen, frameLen-frameHop, 'nodelay');
    
    %===WINDOWING====
    win      = hamming(frameLen);
    nFrames  = size(frames,2);
    framesW  = frames .* win;

    fig4 = figure('Name','Tahap 4 - Framing dan Windowing','NumberTitle','off', ...
                  'Position',[50 600 1400 500]);
    nShow = min(5, nFrames);
    subplot(2, nShow+1, 1:(nShow+1));
    plot(tSeg, seg,'Color',[0.3 0.3 0.8],'LineWidth',0.7); hold on;
    colors4 = winter(nShow);
    for k = 1:nShow
        idxS = (k-1)*frameHop + 1;
        idxE = idxS + frameLen - 1;
        if idxE <= segSamples
            xS = (idxS-1)/fs; xE = (idxE-1)/fs;
            patch([xS xE xE xS],[0.8 0.8 -0.8 -0.8],colors4(k,:), ...
                  'FaceAlpha',0.15,'EdgeColor',colors4(k,:),'LineWidth',1.2);
        end
    end
    title(sprintf('Segmen dengan %d Frame Pertama  (frame = %.0fms, hop = %.0fms)', ...
          nShow, cfg.frameLen*1000, cfg.frameHop*1000));
    xlabel('Waktu (s)'); ylabel('Amplitudo'); grid on; hold off;

    for k = 1:nShow
        subplot(2, nShow+1, nShow+1+k);
        tF = (0:frameLen-1)/fs*1000;
        plot(tF, frames(:,k),'Color',[0.7 0.7 0.7],'LineWidth',0.8); hold on;
        plot(tF, framesW(:,k),'Color',colors4(k,:),'LineWidth',1.2);
        if k==1, legend('Sebelum','Sesudah','FontSize',7); end
        title(sprintf('Frame %d',k),'FontSize',9);
        xlabel('ms'); grid on; hold off;
    end
    subplot(2,nShow+1,2*(nShow+1));
    tW = (0:frameLen-1)/fs*1000;
    plot(tW, win,'k-','LineWidth',1.5);
    title('Window Hamming','FontSize',9); xlabel('ms'); grid on;
    sgtitle(sprintf('TAHAP 4: Framing (%dms) + Windowing Hamming  |  %d frame total', ...
            round(cfg.frameLen*1000), nFrames),'FontSize',12,'FontWeight','bold');

    %% ============================================================
    %%  TAHAP 5: FFT (Spektrum)
    %% ============================================================
    fprintf('[5/14] FFT...\n');
    nFFT  = 2^nextpow2(frameLen);
    freqV = (0:nFFT/2) * fs / nFFT;

    fig5 = figure('Name','Tahap 5 - FFT','NumberTitle','off', ...
                  'Position',[50 600 1400 500]);
    nShowFFT = min(4, nFrames);
    for k = 1:nShowFFT
        subplot(2, nShowFFT, k);
        plot((0:frameLen-1)/fs*1000, framesW(:,k),'Color',[0.3 0.5 0.8],'LineWidth',0.8);
        title(sprintf('Frame %d - Domain Waktu',k),'FontSize',9);
        xlabel('ms'); ylabel('Amp'); grid on;

        subplot(2, nShowFFT, nShowFFT+k);
        magSpec = abs(fft(framesW(:,k), nFFT));
        magSpec = magSpec(1:nFFT/2+1);
        plot(freqV/1000, 20*log10(magSpec+eps),'Color',[0.8 0.3 0.2],'LineWidth',1);
        title(sprintf('Frame %d - Domain Frekuensi (FFT)',k),'FontSize',9);
        xlabel('Frekuensi (kHz)'); ylabel('Magnitude (dB)'); grid on;
        xlim([0 fs/2000]);
    end
    sgtitle(sprintf('TAHAP 5: Fast Fourier Transform (FFT)  |  nFFT = %d  |  Resolusi = %.1f Hz', ...
            nFFT, fs/nFFT),'FontSize',12,'FontWeight','bold');

    %% ============================================================
    %%  TAHAP 6: Ekstraksi Fitur Spektral Dasar
    %% ============================================================
    fprintf('[6/14] Fitur spektral dasar...\n');
    logEn = zeros(1,nFrames); specCentroid = zeros(1,nFrames);
    specBandwidth = zeros(1,nFrames); specRolloff = zeros(1,nFrames);
    specFlux = zeros(1,nFrames); zcr = zeros(1,nFrames);
    rmse_v = zeros(1,nFrames); crestFactor = zeros(1,nFrames);
    specFlat = zeros(1,nFrames); specIrreg = zeros(1,nFrames);
    prevMag = zeros(nFFT/2+1, 1); nBins = nFFT/2+1;

    for i = 1:nFrames
        fr = frames(:,i) .* win;
        magFull = abs(fft(fr, nFFT));
        mag = magFull(1:nBins); powSpec = mag.^2;
        magSum = sum(mag) + eps;
        en = sum(fr.^2);
        logEn(i) = log(en + eps);
        rmse_v(i) = sqrt(mean(fr.^2));
        zcr(i) = sum(abs(diff(sign(fr)))) / (2*frameLen);
        crestFactor(i) = (max(abs(fr))+eps)/(rmse_v(i)+eps);
        specCentroid(i) = sum(freqV .* mag') / magSum;
        specBandwidth(i)= sqrt(sum(((freqV-specCentroid(i)).^2).*mag')/magSum);
        cumS = cumsum(mag); rollIdx = find(cumS>=0.85*cumS(end),1);
        if isempty(rollIdx), rollIdx=numel(freqV); end
        specRolloff(i) = freqV(rollIdx);
        specFlux(i) = sum((mag - prevMag).^2); prevMag = mag;
        geoMean = exp(mean(log(powSpec+eps)));
        ariMean = mean(powSpec)+eps;
        specFlat(i) = geoMean/ariMean;
        if nBins>2
            midB = 2:nBins-1;
            specIrreg(i) = sum(abs(mag(midB)-(mag(midB-1)+mag(midB+1))/2))/(magSum+eps);
        end
    end
    tFr = (0:nFrames-1)*frameHop/fs;

    fig6 = figure('Name','Tahap 6 - Fitur Spektral Dasar','NumberTitle','off', ...
                  'Position',[50 600 1400 600]);
    subplot(3,3,1); plot(tFr,logEn,'b-','LineWidth',1); title('Log Energy'); xlabel('s'); grid on;
    subplot(3,3,2); plot(tFr,specCentroid/1000,'r-','LineWidth',1); title('Spectral Centroid (kHz)'); xlabel('s'); grid on;
    subplot(3,3,3); plot(tFr,specBandwidth/1000,'m-','LineWidth',1); title('Spectral Bandwidth (kHz)'); xlabel('s'); grid on;
    subplot(3,3,4); plot(tFr,specRolloff/1000,'g-','LineWidth',1); title('Spectral Rolloff (kHz)'); xlabel('s'); grid on;
    subplot(3,3,5); plot(tFr,specFlux,'Color',[0.5 0.2 0.8],'LineWidth',1); title('Spectral Flux'); xlabel('s'); grid on;
    subplot(3,3,6); plot(tFr,zcr,'Color',[0.9 0.5 0.1],'LineWidth',1); title('Zero Crossing Rate'); xlabel('s'); grid on;
    subplot(3,3,7); plot(tFr,rmse_v,'Color',[0.2 0.7 0.5],'LineWidth',1); title('RMSE'); xlabel('s'); grid on;
    subplot(3,3,8); plot(tFr,crestFactor,'Color',[0.7 0.1 0.5],'LineWidth',1); title('Crest Factor'); xlabel('s'); grid on;
    subplot(3,3,9); plot(tFr,specFlat,'k-','LineWidth',1); title('Spectral Flatness'); xlabel('s'); grid on;
    sgtitle('TAHAP 6: Ekstraksi Fitur Spektral Dasar (Per Frame)', ...
            'FontSize',12,'FontWeight','bold');

    %% ============================================================
    %%  TAHAP 7: MFCC
    %% ============================================================
    fprintf('[7/14] MFCC...\n');
    nMFCC_vis = 20;
    coeffs = mfcc(seg, fs, ...
        'NumCoeffs',     nMFCC_vis, ...
        'WindowLength',  frameLen, ...
        'OverlapLength', frameLen - frameHop, ...
        'FFTLength',     nFFT);

    delta1 = hitungDelta(coeffs);
    delta2 = hitungDelta(delta1);

    fig7 = figure('Name','Tahap 7 - MFCC','NumberTitle','off', ...
                  'Position',[50 600 1400 600]);
    subplot(3,2,1);
    imagesc(coeffs'); colorbar; colormap(gca,'jet');
    title(sprintf('MFCC (%d Koef) - Heatmap',nMFCC_vis));
    xlabel('Frame'); ylabel('Koefisien MFCC'); set(gca,'YDir','normal');

    subplot(3,2,2);
    mfccMean = mean(coeffs,1); mfccStd = std(coeffs,0,1);
    nCoeffActual = numel(mfccMean);
    errorbar(1:nCoeffActual, mfccMean, mfccStd, 'bo-','LineWidth',1.2,'MarkerSize',5);
    title(sprintf('MFCC: Mean ± Std per Koefisien (%d koef)',nCoeffActual)); xlabel('Koefisien'); ylabel('Nilai'); grid on;

    subplot(3,2,3);
    imagesc(delta1'); colorbar; colormap(gca,'cool');
    title('Delta MFCC (Δ)'); xlabel('Frame'); ylabel('Koefisien'); set(gca,'YDir','normal');

    subplot(3,2,4);
    imagesc(delta2'); colorbar; colormap(gca,'hot');
    title('Delta-Delta MFCC (ΔΔ)'); xlabel('Frame'); ylabel('Koefisien'); set(gca,'YDir','normal');

    subplot(3,2,5);
    tMFCC = (0:size(coeffs,1)-1)*frameHop/fs;
    plot(tMFCC, coeffs(:,1),'b-','LineWidth',1); hold on;
    plot(tMFCC, coeffs(:,2),'r-','LineWidth',1);
    plot(tMFCC, coeffs(:,3),'g-','LineWidth',1);
    legend('MFCC-1','MFCC-2','MFCC-3'); title('MFCC 1-3 vs Waktu');
    xlabel('Waktu (s)'); grid on; hold off;

    subplot(3,2,6);
    boxplot(coeffs,'Notch','on');
    title('Distribusi Setiap Koefisien MFCC (Boxplot)');
    xlabel('Koefisien MFCC'); ylabel('Nilai');
    sgtitle('TAHAP 7: Mel Frequency Cepstral Coefficients (MFCC)', ...
            'FontSize',12,'FontWeight','bold');

    %% ============================================================
    %%  TAHAP 8: Struktur Harmonik
    %% ============================================================
    fprintf('[8/14] Struktur harmonik...\n');
    nHarm = 8;
    harmPow = zeros(nHarm, nFrames);
    f0Vals  = zeros(1, nFrames);
    ampVals = zeros(1, nFrames);
    hnrVals = zeros(1, nFrames);

    for i = 1:nFrames
        fr = frames(:,i) .* win;
        magFull = abs(fft(fr, nFFT));
        mag = magFull(1:nBins); powSpec = mag.^2;
        r = xcorr(fr,'coeff'); r = r(frameLen:end);
        minL = round(fs/800); maxL = min(round(fs/50),frameLen-1);
        if minL < maxL
            [peakV, peakIdx] = max(r(minL+1:maxL+1));
            f0Est = fs/(peakIdx+minL);
            f0Vals(i) = f0Est; ampVals(i) = peakV;
            for h = 1:nHarm
                hFreq = f0Est*h;
                if hFreq < fs/2
                    hBin = round(hFreq*nFFT/fs)+1;
                    hBin = max(1,min(nBins,hBin));
                    binR = max(1,hBin-2):min(nBins,hBin+2);
                    harmPow(h,i) = max(powSpec(binR));
                end
            end
        end
        hnrVals(i) = hitungHNR(fr,fs);
    end
    harmMean = mean(harmPow,2)';
    harmStd  = std(harmPow,0,2)';

    fig8 = figure('Name','Tahap 8 - Struktur Harmonik','NumberTitle','off', ...
                  'Position',[50 600 1400 550]);
    subplot(2,3,1);
    bar(1:nHarm, harmMean, 'FaceColor',[0.3 0.6 0.9]); hold on;
    errorbar(1:nHarm, harmMean, harmStd,'k.','LineWidth',1.2);
    title('Rata-rata Power per Harmonik'); xlabel('Harmonik ke-'); ylabel('Power'); grid on; hold off;

    subplot(2,3,2);
    imagesc(harmPow); colorbar; colormap(gca,'parula');
    title('Harmonic Power Map (Harmonik x Frame)');
    xlabel('Frame'); ylabel('Harmonik ke-'); set(gca,'YDir','normal');

    subplot(2,3,3);
    validF0 = f0Vals(f0Vals>0);
    if ~isempty(validF0)
        plot(tFr(f0Vals>0), validF0,'r.-','LineWidth',1,'MarkerSize',8);
    else
        text(0.5,0.5,'F0 tidak terdeteksi','HorizontalAlignment','center','Units','normalized');
    end
    title('F0 (Fundamental Frequency) Trajectory'); xlabel('Waktu (s)'); ylabel('Hz'); grid on;

    subplot(2,3,4);
    plot(tFr, hnrVals,'g-','LineWidth',1.2);
    title('HNR (Harmonics-to-Noise Ratio)'); xlabel('Waktu (s)'); ylabel('dB'); grid on;

    subplot(2,3,5);
    oddIdx=[1,3,5,7]; evenIdx=[2,4,6,8];
    oddPow=sum(harmMean(oddIdx)); evenPow=sum(harmMean(evenIdx));
    bar([oddPow, evenPow],'FaceColor',[0.8 0.3 0.3]);
    set(gca,'XTickLabel',{'Harmonik Ganjil','Harmonik Genap'});
    title(sprintf('Rasio Ganjil/Genap = %.2f',oddPow/(evenPow+eps)));
    ylabel('Total Power'); grid on;

    subplot(2,3,6);
    if all(harmMean>0)
        p = polyfit(1:nHarm, log(harmMean+eps), 1);
        plot(1:nHarm, log(harmMean+eps),'bo-','LineWidth',1.5,'MarkerSize',7); hold on;
        plot(1:nHarm, polyval(p,1:nHarm),'r--','LineWidth',1.5);
        legend('Log Power','Fit Linear'); hold off;
        title(sprintf('Harmonik Decay Slope = %.3f',-p(1)));
    else
        bar(1:nHarm, harmMean+eps);
        title('Log Harmonic Power');
    end
    xlabel('Harmonik ke-'); ylabel('Log Power'); grid on;
    sgtitle('TAHAP 8: Analisis Struktur Harmonik (HNR, F0, Harmonic Power)', ...
            'FontSize',12,'FontWeight','bold');

    %% ============================================================
    %%  TAHAP 9: Chain Modulation Analysis
    %% ============================================================
    fprintf('[9/14] Chain modulation...\n');
    envelope    = abs(hilbert(seg));
    envNorm     = envelope / (mean(envelope)+eps);
    nFFT_env    = 2048;
    [envPxx, envF] = pwelch(envNorm, round(fs/2), round(fs/4), nFFT_env, fs);

    idxChainAM = envF>=15 & envF<=80;
    idxMotorAM = envF>=3  & envF<=15;
    idxAMTot   = envF>=1  & envF<=150;
    totAMPow   = sum(envPxx(idxAMTot))+eps;
    chainAMEn  = sum(envPxx(idxChainAM))/totAMPow;
    motorAMEn  = sum(envPxx(idxMotorAM))/totAMPow;

    minLagEnv = round(fs/80); maxLagEnv = round(fs/15);
    [rEnv, lagEnv] = xcorr(envNorm-mean(envNorm), round(fs*0.5),'coeff');
    rEnvPos = rEnv(lagEnv>=0);

    fig9 = figure('Name','Tahap 9 - Chain Modulation','NumberTitle','off', ...
                  'Position',[50 600 1400 600]);
    subplot(2,3,1);
    plot(tSeg, seg,'Color',[0.7 0.7 0.7],'LineWidth',0.5); hold on;
    plot(tSeg, envNorm,'r-','LineWidth',1.2);
    legend('Sinyal','Envelope'); title('Sinyal + Envelope (Hilbert)');
    xlabel('Waktu (s)'); ylabel('Amplitudo'); grid on; hold off;

    subplot(2,3,2);
    plot(envF, 10*log10(envPxx+eps),'b-','LineWidth',1.2); hold on;
    fill([15 80 80 15],[-60 -60 max(10*log10(envPxx+eps)) max(10*log10(envPxx+eps))], ...
         'r','FaceAlpha',0.15,'EdgeColor','r','LineStyle','--');
    fill([3 15 15 3],[-60 -60 max(10*log10(envPxx+eps)) max(10*log10(envPxx+eps))], ...
         'g','FaceAlpha',0.15,'EdgeColor','g','LineStyle','--');
    legend('PSD Envelope','Chain AM (15-80Hz)','Motor AM (3-15Hz)');
    title('PSD Envelope - Deteksi Modulasi Amplitudo'); hold off;
    xlabel('Frekuensi (Hz)'); ylabel('Power (dB)'); xlim([0 150]); grid on;

    subplot(2,3,3);
    lagSec = (0:length(rEnvPos)-1)/fs;
    plot(lagSec, rEnvPos,'Color',[0.5 0.2 0.8],'LineWidth',1.2); hold on;
    if minLagEnv<maxLagEnv && maxLagEnv<length(rEnvPos)
        [peakEnvV,peakEnvI] = max(rEnvPos(minLagEnv+1:maxLagEnv+1));
        peakEnvLag = (peakEnvI+minLagEnv-1)/fs;
        plot(peakEnvLag, peakEnvV,'ro','MarkerSize',10,'LineWidth',2);
        title(sprintf('Autokorelasi Envelope  |  Peak @ %.3f s', peakEnvLag));
    else
        title('Autokorelasi Envelope');
    end
    xlabel('Lag (s)'); ylabel('Korelasi'); xlim([0 0.5]); grid on; hold off;

    subplot(2,3,4);
    barData = [chainAMEn, motorAMEn, 1-chainAMEn-motorAMEn];
    barData(barData<0) = 0;
    bar(barData,'FaceColor','flat','CData',[0.9 0.2 0.2; 0.2 0.7 0.2; 0.5 0.5 0.8]);
    set(gca,'XTickLabel',{'Chain AM (15-80Hz)','Motor AM (3-15Hz)','Lainnya'});
    title(sprintf('Distribusi Energi AM\nRasio Chain/Motor = %.2f', chainAMEn/(motorAMEn+eps)));
    ylabel('Proporsi Energi'); grid on;

    subplot(2,3,5);
    envCrest = max(envNorm)/(mean(envNorm)+eps);
    envelopeVar  = var(envelope);
    envelopeSkew = skewness(envelope);
    envelopeKurt = kurtosis(envelope);
    barNames = {'Crest Factor','Var (x10)','Skewness','Kurtosis'};
    barVals  = [envCrest, envelopeVar*10, envelopeSkew, envelopeKurt];
    bar(barVals,'FaceColor',[0.2 0.6 0.9]);
    set(gca,'XTickLabel',barNames); title('Statistik Envelope');
    ylabel('Nilai'); grid on;

    subplot(2,3,6);
    plot(tSeg, envNorm,'b-','LineWidth',0.7); hold on;
    meanEnv = mean(envNorm); stdEnv = std(envNorm);
    yline(meanEnv,'r--','LineWidth',1.5,'Label','Mean');
    yline(meanEnv+stdEnv,'g--','LineWidth',1,'Label','Mean+Std');
    title('Envelope Ternormalisasi dengan Statistik');
    xlabel('Waktu (s)'); ylabel('Amplitudo'); grid on; hold off;
    sgtitle('TAHAP 9: Chain Modulation Analysis (AM 15-80 Hz)', ...
            'FontSize',12,'FontWeight','bold');

    %% ============================================================
    %%  TAHAP 10: Fitur Lanjutan (Dipisahkan)
    %% ============================================================
    fprintf('[10/14] Fitur lanjutan (7 subplot)...\n');

    %% --- Spectral Roughness ---
    specIrregMean = mean(specIrreg); specIrregStd = std(specIrreg);
    specFlatMean  = mean(specFlat);  specFlatStd  = std(specFlat);

    %% --- Transient ---
    crestMean = mean(crestFactor); crestStd = std(crestFactor);
    fluxThresh   = mean(specFlux)+std(specFlux);
    onsetDensity = sum(specFlux>fluxThresh)/nFrames;
    onsetFrames  = find(specFlux>fluxThresh);

    %% --- Sub-band Temporal Dynamics ---
    nMelBand = 40;
    melFilters = hitungMelFilterbank(nMelBand, nFFT, fs);
    lmfeFrames = zeros(nMelBand, nFrames);
    for i = 1:nFrames
        fr = frames(:,i).*win; magFull=abs(fft(fr,nFFT));
        powSpec=magFull(1:nBins).^2;
        melE=melFilters*powSpec; lmfeFrames(:,i)=log(melE+eps);
    end
    nSB=8; subBandEdges=[0,150,300,600,1200,2400,4000,6000,fs/2];
    subBandEdges(subBandEdges>fs/2)=fs/2;
    [pxx,fPxx]=pwelch(seg,frameLen,frameLen-frameHop,nFFT,fs);
    totPow=sum(pxx)+eps; sbEnergy=zeros(1,nSB);
    for sb=1:nSB
        idxSB=fPxx>=subBandEdges(sb)&fPxx<subBandEdges(sb+1);
        sbEnergy(sb)=sum(pxx(idxSB))/totPow;
    end

    %% --- LMFE ---
    lmfeMean=mean(lmfeFrames,2)'; lmfeStd=std(lmfeFrames,0,2)';
    lmfeGrad=mean(abs(diff(lmfeFrames,1,2)),2)';

    %% --- Spectral Contrast ---
    nSC=8; scEdges=[0,150,300,600,1200,2400,4800,7200,fs/2];
    scEdges(scEdges>fs/2)=fs/2;
    scPeak=zeros(nSC,nFrames); scValley=zeros(nSC,nFrames);
    for i=1:nFrames
        fr=frames(:,i).*win; magFull=abs(fft(fr,nFFT)); mag=magFull(1:nBins);
        for sb=1:nSC
            binL=max(1,round(scEdges(sb)*nFFT/fs)+1);
            binH=min(nBins,round(scEdges(sb+1)*nFFT/fs)+1);
            if binH>binL
                subM=mag(binL:binH); nN=max(1,round(0.02*length(subM)));
                sortedS=sort(subM,'descend');
                scPeak(sb,i)=mean(sortedS(1:min(nN,end)));
                scValley(sb,i)=mean(sortedS(max(1,end-nN+1):end));
            end
        end
    end
    scContrast=log(scPeak./(scValley+eps));
    scContrastMean=mean(scContrast,2)'; scContrastStd=std(scContrast,0,2)';

    %% --- CPP ---
    cepstrum=real(ifft(log(abs(fft(seg,nFFT)).^2+eps)));
    cepstrum=cepstrum(1:nFFT/2);
    minQ=round(fs/800); maxQ=min(round(fs/50),length(cepstrum)-1);
    if minQ<maxQ
        [cppPeakVal,cppPeakIdx]=max(cepstrum(minQ+1:maxQ+1));
        cppPeakIdx=cppPeakIdx+minQ;
        qRange=(minQ:maxQ)'; p_cpp=polyfit(qRange,cepstrum(minQ+1:maxQ+1),1);
        baseline=polyval(p_cpp,cppPeakIdx); cpp=cppPeakVal-baseline;
    else, cpp=0; cppPeakIdx=1; end

    %% --- F0 Trajectory ---
    validF0=f0Vals(f0Vals>0); validAmp=ampVals(ampVals>0);
    if numel(validF0)>2
        f0Jitter=mean(abs(diff(validF0)))/(mean(validF0)+eps);
        f0RJitter=std(validF0)/(mean(validF0)+eps);
    else, f0Jitter=0; f0RJitter=0; end
    if numel(validAmp)>2
        shimmer=mean(abs(diff(validAmp)))/(mean(validAmp)+eps);
    else, shimmer=0; end

    fig10 = figure('Name','Tahap 10 - Fitur Lanjutan','NumberTitle','off', ...
                   'Position',[50 600 1500 700]);

    % (a) Spectral Roughness
    subplot(3,3,1);
    plot(tFr,specIrreg,'b-','LineWidth',1); hold on;
    yline(specIrregMean,'r--','LineWidth',1.5,'Label','Mean');
    title(sprintf('Spectral Roughness/Irregularity\nMean=%.4f Std=%.4f',specIrregMean,specIrregStd));
    xlabel('Waktu (s)'); grid on; hold off;

    subplot(3,3,2);
    plot(tFr,specFlat,'g-','LineWidth',1); hold on;
    yline(specFlatMean,'r--','LineWidth',1.5,'Label','Mean');
    title(sprintf('Spectral Flatness\nMean=%.4f Std=%.4f',specFlatMean,specFlatStd));
    xlabel('Waktu (s)'); grid on; hold off;

    % (b) Transient
    subplot(3,3,3);
    plot(tFr,specFlux,'Color',[0.5 0.0 0.7],'LineWidth',1); hold on;
    if ~isempty(onsetFrames)
        plot(tFr(onsetFrames),specFlux(onsetFrames),'rv','MarkerSize',8,'MarkerFaceColor','r');
    end
    yline(fluxThresh,'r--','LineWidth',1,'Label','Threshold');
    title(sprintf('Transient Detection (Onset)\nCrest=%.2f Onset Density=%.3f',crestMean,onsetDensity));
    xlabel('Waktu (s)'); legend('Flux','Onset','FontSize',7); grid on; hold off;

    % (c) Sub-band
    subplot(3,3,4);
    barh(sbEnergy,'FaceColor',[0.3 0.7 0.5],'EdgeColor','w');
    set(gca,'YTickLabel',{'0-150Hz','150-300','300-600','600-1.2k','1.2-2.4k','2.4-4k','4-6k','6k-Nyq'});
    title('Sub-band Energy Distribution'); xlabel('Energi Relatif'); grid on;

    % (d) LMFE
    subplot(3,3,5);
    imagesc(lmfeFrames); colorbar; colormap(gca,'jet');
    title('Log Mel Filter Bank Energy (40 Band)');
    xlabel('Frame'); ylabel('Mel Band'); set(gca,'YDir','normal');

    % (e) Spectral Contrast
    subplot(3,3,6);
    errorbar(1:nSC, scContrastMean, scContrastStd,'bo-','LineWidth',1.5,'MarkerSize',7);
    title('Spectral Contrast (8 Band)  Mean ± Std');
    xlabel('Sub-band'); ylabel('Log Kontras'); grid on;

    % (f) CPP
    subplot(3,3,7);
    quefrencyAxis = (0:length(cepstrum)-1)/fs;
    plot(quefrencyAxis*1000, cepstrum,'b-','LineWidth',0.8); hold on;
    if cppPeakIdx>=1 && cppPeakIdx<=length(cepstrum)
        plot(cppPeakIdx/fs*1000, cepstrum(cppPeakIdx),'rv','MarkerSize',12,'MarkerFaceColor','r');
    end
    title(sprintf('Cepstral Peak Prominence (CPP)\nCPP = %.4f',cpp));
    xlabel('Quefrency (ms)'); ylabel('Cepstrum'); xlim([0 25]); grid on; hold off;

    % (g) F0 Trajectory
    subplot(3,3,8);
    tFrIdx = (0:nFrames-1)*frameHop/fs;
    if any(f0Vals>0)
        f0Plot = f0Vals; f0Plot(f0Plot==0) = NaN;
        plot(tFrIdx, f0Plot,'m.-','LineWidth',1,'MarkerSize',10);
        title(sprintf('F0 Trajectory\nJitter=%.4f  Shimmer=%.4f',f0Jitter,shimmer));
    else
        text(0.5,0.5,'F0 tidak terdeteksi','HorizontalAlignment','center','Units','normalized');
        title('F0 Trajectory'); 
    end
    xlabel('Waktu (s)'); ylabel('F0 (Hz)'); grid on;

    subplot(3,3,9);
    metricNames = {'Spectral Irreg','Spec Flat','Crest Mean','Onset Dens','CPP','Jitter','Shimmer'};
    metricVals  = [specIrregMean, specFlatMean, crestMean, onsetDensity, cpp, f0Jitter, shimmer];
    metricValsN = (metricVals - min(metricVals)) ./ (max(metricVals)-min(metricVals)+eps);
    barh(metricValsN,'FaceColor',[0.8 0.4 0.1],'EdgeColor','w');
    set(gca,'YTickLabel',metricNames,'FontSize',8);
    title('Ringkasan Fitur Lanjutan (Ternormalisasi)'); xlabel('Nilai Ternormalisasi'); grid on;

    sgtitle('TAHAP 10: Spectral Roughness | Transient | Sub-band | LMFE | SC | CPP | F0', ...
            'FontSize',11,'FontWeight','bold');

    %% ============================================================
    %%  TAHAP 11: Ekstrak Fitur Lengkap & Penyebaran
    %% ============================================================
    fprintf('[11/14] Penyebaran fitur...\n');

    %% Kumpulkan fitur dari semua segmen semua kelas
    dataFolders11 = {};
    baseKelas = {'chainsaw', 'non_chainsaw_mirip', 'non_chainsaw_tidak_mirip'};
    for bk = 1:numel(baseKelas)
        basePath = fullfile(cfg.datasetRoot, baseKelas{bk});
        if ~isfolder(basePath), continue; end
        subFolders = dir(basePath);
        subFolders = subFolders([subFolders.isdir] & ~startsWith({subFolders.name},'.'));
        for sf = 1:numel(subFolders)
            fullSub = fullfile(basePath, subFolders(sf).name);
            label   = sprintf('%s/%s', baseKelas{bk}, subFolders(sf).name);
            dataFolders11(end+1,:) = {fullSub, label}; %#ok<AGROW>
        end
    end
    allFeat11=[]; allKelas11={};
    for di=1:size(dataFolders11,1)
        fp11=dataFolders11{di,1}; nm11=dataFolders11{di,2};
        if ~isfolder(fp11), continue; end
        ff11=[dir(fullfile(fp11,'*.wav'));dir(fullfile(fp11,'*.mp3'))];
        for fi=1:min(numel(ff11),3)
            try
                [ytmp,fstmp]=audioread(fullfile(ff11(fi).folder,ff11(fi).name));
                ytmp=preprocessAudio(ytmp,fstmp,cfg);
                if isempty(ytmp), continue; end
                st11=1:hopSamples:(length(ytmp)-segSamples+1);
                for ki=1:numel(st11)
                    sg11=ytmp(st11(ki):st11(ki)+segSamples-1);
                    f11=ekstrakFitur(sg11,cfg.targetFs,cfg);
                    if ~isempty(f11)
                        allFeat11=[allFeat11;f11]; %#ok<AGROW>
                        allKelas11{end+1}=nm11;    %#ok<SAGROW>
                    end
                end
            catch, end
        end
    end

    fig11 = figure('Name','Tahap 11 - Penyebaran Fitur','NumberTitle','off', ...
                   'Position',[50 600 1400 600]);
    if size(allFeat11,1)>3
        %% PCA 3D
        mu11=mean(allFeat11,1); sig11=std(allFeat11,0,1); sig11(sig11==0)=1;
        fnorm11=(allFeat11-mu11)./sig11;
        fnorm11(~isfinite(fnorm11))=0;
        [~,sc11]=pca(fnorm11);
        subplot(1,2,1);
        unikKel=unique(allKelas11,'stable');
        clrs11=lines(numel(unikKel));
        hold on;
        for ci=1:numel(unikKel)
            idx11=strcmp(allKelas11,unikKel{ci});
            scatter(sc11(idx11,1),sc11(idx11,2),40,clrs11(ci,:),'filled','DisplayName',unikKel{ci});
        end
        legend('Location','best','FontSize',8); xlabel('PC1'); ylabel('PC2');
        title('PCA 2D - Penyebaran Fitur per Kelas'); grid on; hold off;

        subplot(1,2,2);
        if size(sc11,2)>=3
            hold on;
            for ci=1:numel(unikKel)
                idx11=strcmp(allKelas11,unikKel{ci});
                scatter3(sc11(idx11,1),sc11(idx11,2),sc11(idx11,3),40,clrs11(ci,:),'filled','DisplayName',unikKel{ci});
            end
            legend('Location','best','FontSize',8);
            xlabel('PC1'); ylabel('PC2'); zlabel('PC3');
            title('PCA 3D - Penyebaran Fitur per Kelas'); grid on; view(30,20); hold off;
        end
    else
        text(0.5,0.5,'Data tidak cukup untuk visualisasi PCA','HorizontalAlignment','center','Units','normalized');
    end
    sgtitle('TAHAP 11: Penyebaran Fitur Setelah Ekstraksi (PCA)', ...
            'FontSize',12,'FontWeight','bold');

    %% ============================================================
    %%  TAHAP 12: Before & After Normalisasi Fitur
    %% ============================================================
    fprintf('[12/14] Normalisasi fitur...\n');
    if size(allFeat11,1)>1
        mu12=mean(allFeat11,1); sig12=std(allFeat11,0,1); sig12(sig12==0)=1;
        featNorm12=(allFeat11-mu12)./sig12;
        featNorm12(~isfinite(featNorm12))=0;

        fig12=figure('Name','Tahap 12 - Normalisasi Fitur','NumberTitle','off', ...
                     'Position',[50 600 1400 600]);
        nShowFeat=min(10,size(allFeat11,2));

        subplot(2,3,1);
        boxplot(allFeat11(:,1:nShowFeat));
        title('SEBELUM Normalisasi: Boxplot (10 Fitur Pertama)');
        xlabel('Indeks Fitur'); ylabel('Nilai Asli'); grid on;

        subplot(2,3,2);
        boxplot(featNorm12(:,1:nShowFeat));
        title('SETELAH Normalisasi Z-score: Boxplot');
        xlabel('Indeks Fitur'); ylabel('Nilai Ternormalisasi'); grid on;

        subplot(2,3,3);
        muV=mean(allFeat11,1); muN=mean(featNorm12,1);
        sigV=std(allFeat11,0,1); sigN=std(featNorm12,0,1);
        plot(muV(1:min(50,end)),'b-','LineWidth',1.2,'DisplayName','Before Mean'); hold on;
        plot(sigV(1:min(50,end)),'b--','LineWidth',1,'DisplayName','Before Std');
        plot(muN(1:min(50,end)),'r-','LineWidth',1.2,'DisplayName','After Mean');
        plot(sigN(1:min(50,end)),'r--','LineWidth',1,'DisplayName','After Std');
        legend('FontSize',8); title('Mean & Std: Before vs After (50 Fitur)');
        xlabel('Indeks Fitur'); grid on; hold off;

        subplot(2,3,4);
        histogram(allFeat11(:),'NumBins',60,'FaceColor',[0.3 0.5 0.9],'EdgeColor','none');
        title('Distribusi Semua Nilai Fitur (SEBELUM)');
        xlabel('Nilai'); ylabel('Count'); grid on;

        subplot(2,3,5);
        histogram(featNorm12(:),'NumBins',60,'FaceColor',[0.9 0.3 0.3],'EdgeColor','none');
        title('Distribusi Semua Nilai Fitur (SETELAH)');
        xlabel('Nilai Z-score'); ylabel('Count'); grid on;

        subplot(2,3,6);
        rangeBefore=max(allFeat11)-min(allFeat11);
        rangeAfter=max(featNorm12)-min(featNorm12);
        scatter(rangeBefore(1:min(100,end)),rangeAfter(1:min(100,end)),30,'filled');
        xlabel('Range Sebelum'); ylabel('Range Sesudah'); grid on;
        title('Range Fitur: Sebelum vs Sesudah Normalisasi');

        sgtitle('TAHAP 12: Before & After Normalisasi Fitur (Z-score)', ...
                'FontSize',12,'FontWeight','bold');
    else
        fig12=figure('Name','Tahap 12','NumberTitle','off');
        text(0.5,0.5,'Data tidak cukup','HorizontalAlignment','center','Units','normalized');
    end

    %% ============================================================
    %%  TAHAP 13: Grid Search SVM
    %% ============================================================
    fprintf('[13/14] Grid Search SVM (visualisasi)...\n');
    if size(allFeat11,1)>5 && numel(unique(cellfun(@(x)strcmp(x,'CS-Dekat')||strcmp(x,'CS-Jauh'),allKelas11)))>0
        labels13=cellfun(@(x) double(strcmp(x,'CS-Dekat')||strcmp(x,'CS-Jauh')), allKelas11(:));
        mu13=mean(allFeat11,1); sig13=std(allFeat11,0,1); sig13(sig13==0)=1;
        feat13=(allFeat11-mu13)./sig13; feat13(~isfinite(feat13))=0;

        cValues13=[0.1,1,10,100]; gammaValues13=[0.001,0.01,0.1,1];
        accGrid13=zeros(numel(cValues13),numel(gammaValues13));

        fprintf('     Menjalankan Grid Search...\n');
        for ci=1:numel(cValues13)
            for gi=1:numel(gammaValues13)
                try
                    svmTmp=fitcsvm(feat13,labels13,'KernelFunction','rbf', ...
                        'BoxConstraint',cValues13(ci), ...
                        'KernelScale',1/gammaValues13(gi), ...
                        'Standardize',false,'CrossVal','on','KFold',3);
                    accGrid13(ci,gi)=(1-kfoldLoss(svmTmp))*100;
                catch, accGrid13(ci,gi)=0; end
            end
        end

        fig13=figure('Name','Tahap 13 - Grid Search SVM','NumberTitle','off', ...
                     'Position',[50 600 1400 550]);
        subplot(1,3,1);
        imagesc(accGrid13);
        set(gca,'XTick',1:4,'XTickLabel',{'0.001','0.01','0.1','1'},'XTickLabelRotation',45);
        set(gca,'YTick',1:4,'YTickLabel',{'0.1','1','10','100'});
        xlabel('Gamma'); ylabel('C (BoxConstraint)'); colorbar;
        title('Grid Search Accuracy Map (%)'); colormap(gca,'hot');
        for ci=1:4, for gi=1:4
            text(gi,ci,sprintf('%.1f',accGrid13(ci,gi)),'HorizontalAlignment','center', ...
                 'Color','w','FontSize',9,'FontWeight','bold');
        end; end

        subplot(1,3,2);
        [bestAcc13,bestIdx]=max(accGrid13(:));
        [bC13,bG13]=ind2sub(size(accGrid13),bestIdx);
        plot(accGrid13','LineWidth',1.5,'Marker','o');
        legend(cellstr(num2str(cValues13','C=%.1f')),'FontSize',8,'Location','best');
        xlabel('Gamma Indeks'); ylabel('CV Accuracy (%)'); grid on;
        title(sprintf('Accuracy per C vs Gamma\nBest: C=%.1f Gamma=%.3f Acc=%.1f%%', ...
              cValues13(bC13),gammaValues13(bG13),bestAcc13));

        subplot(1,3,3);
        bar3(accGrid13);
        set(gca,'XTickLabel',{'0.001','0.01','0.1','1'}, ...
                'YTickLabel',{'C=0.1','C=1','C=10','C=100'});
        xlabel('Gamma'); ylabel('C'); zlabel('Accuracy (%)');
        title('3D Grid Search Landscape');
        sgtitle('TAHAP 13: Optimasi Hyperparameter SVM (Grid Search)', ...
                'FontSize',12,'FontWeight','bold');
    else
        fig13=figure('Name','Tahap 13 - Grid Search SVM','NumberTitle','off');
        text(0.5,0.5,'Data tidak cukup untuk Grid Search. Jalankan training terlebih dahulu.', ...
             'HorizontalAlignment','center','Units','normalized','FontSize',12);
        title('TAHAP 13: Grid Search SVM');
    end

    %% ============================================================
    %%  TAHAP 14: Hasil Training SVM
    %% ============================================================
    fprintf('[14/14] Hasil training SVM...\n');
    modelFile = 'chainsaw_model_v10_visual.mat';
    fig14=figure('Name','Tahap 14 - Hasil Training SVM','NumberTitle','off', ...
                 'Position',[50 600 1400 600]);

    if exist(modelFile,'file')==2
        load(modelFile,'svmModelProb','mu','sigma','cfg','bestC','bestGamma', ...
             'precision','recall','f1score','specificity','accTrain');

        %% Visualisasi bobot/support vector menggunakan PCA pada fitur yang ada
        if size(allFeat11,1) > 5
            muSV=mean(allFeat11,1); sigSV=std(allFeat11,0,1); sigSV(sigSV==0)=1;
            featSV=(allFeat11-muSV)./sigSV; featSV(~isfinite(featSV))=0;
            labelsSV=cellfun(@(x) double(strcmp(x,'CS-Dekat')||strcmp(x,'CS-Jauh')), allKelas11(:));
            [~,scSV]=pca(featSV);

            subplot(1,3,1);
            hold on;
            unikSV=unique(labelsSV);
            clrSV=[0.2 0.6 0.9;0.9 0.2 0.2];
            for ci=1:numel(unikSV)
                idx=labelsSV==unikSV(ci);
                if unikSV(ci)==0
                    scatter(scSV(idx,1),scSV(idx,2),40,clrSV(1,:),'filled','DisplayName','Non-Chainsaw');
                else
                    scatter(scSV(idx,1),scSV(idx,2),40,clrSV(2,:),'filled','DisplayName','Chainsaw');
                end
            end
            legend('FontSize',9); xlabel('PC1'); ylabel('PC2');
            title('Distribusi Data (PCA 2D)  Train/Test Split'); grid on; hold off;
        end

        subplot(1,3,2);
        metrics14={'Accuracy','Precision','Recall','Specificity','F1-Score'};
        vals14=[accTrain,precision,recall,specificity,f1score];
        b14=bar(vals14,'FaceColor','flat');
        for mi=1:5
            if vals14(mi)>=90, b14.CData(mi,:)=[0.1 0.7 0.2];
            elseif vals14(mi)>=70, b14.CData(mi,:)=[0.9 0.7 0.1];
            else, b14.CData(mi,:)=[0.8 0.1 0.1]; end
            text(mi,vals14(mi)+0.5,sprintf('%.1f%%',vals14(mi)),'HorizontalAlignment','center', ...
                 'FontSize',10,'FontWeight','bold');
        end
        set(gca,'XTickLabel',metrics14,'XTickLabelRotation',20);
        ylim([0 115]); title('Metrik Performa Model SVM');
        ylabel('Nilai (%)'); grid on;
        yline(90,'r--','LineWidth',1.5,'Label','Target 90%');

        subplot(1,3,3); axis off;
        infoStr = {
            '=== INFO MODEL SVM ===', ...
            sprintf('Kernel       : RBF'), ...
            sprintf('Best C       : %.4f', bestC), ...
            sprintf('Best Gamma   : %.5f', bestGamma), ...
            sprintf(''), ...
            '=== PERFORMA TRAINING ===', ...
            sprintf('Accuracy     : %.2f%%', accTrain), ...
            sprintf('Precision    : %.2f%%', precision), ...
            sprintf('Recall       : %.2f%%', recall), ...
            sprintf('Specificity  : %.2f%%', specificity), ...
            sprintf('F1-Score     : %.2f%%', f1score), ...
            sprintf(''), ...
            sprintf('Model: %s', modelFile)
        };
        for li=1:numel(infoStr)
            text(0.05, 1-(li-1)*0.075, infoStr{li}, 'Units','normalized', ...
                 'FontSize',10, 'FontName','FixedWidth', ...
                 'Color', [0.15 0.15 0.5]);
        end
        title('Parameter & Ringkasan Model');
    else
        text(0.5,0.6,'Model belum tersedia.','HorizontalAlignment','center', ...
             'Units','normalized','FontSize',14,'Color',[0.7 0.1 0.1]);
        text(0.5,0.4,'Jalankan opsi 1 (Training) terlebih dahulu.', ...
             'HorizontalAlignment','center','Units','normalized','FontSize',11);
        title('TAHAP 14: Hasil Training SVM');
    end
    sgtitle('TAHAP 14: Hasil Training SVM (Model, Metrik, Distribusi)', ...
            'FontSize',12,'FontWeight','bold');

    %% ============================================================
    %%  RINGKASAN
    %% ============================================================
    fprintf('\n==============================================\n');
    fprintf('  VISUALISASI PIPELINE SELESAI\n');
    fprintf('  14 Figure telah dibuat:\n');
    fprintf('   Fig 1  - Sinyal Audio Mentah\n');
    fprintf('   Fig 2  - Setelah Pre-processing\n');
    fprintf('   Fig 3  - Segmentasi dan Overlap\n');
    fprintf('   Fig 4  - Framing dan Windowing\n');
    fprintf('   Fig 5  - FFT (Domain Frekuensi)\n');
    fprintf('   Fig 6  - Fitur Spektral Dasar\n');
    fprintf('   Fig 7  - MFCC (20 Koef + Delta)\n');
    fprintf('   Fig 8  - Struktur Harmonik\n');
    fprintf('   Fig 9  - Chain Modulation Analysis\n');
    fprintf('   Fig 10 - Fitur Lanjutan (7 Komponen)\n');
    fprintf('   Fig 11 - Penyebaran Fitur (PCA)\n');
    fprintf('   Fig 12 - Before & After Normalisasi\n');
    fprintf('   Fig 13 - Grid Search SVM\n');
    fprintf('   Fig 14 - Hasil Training SVM\n');
    fprintf('==============================================\n');
end


function jalankanVisualisasi()

    cfg = buatKonfigurasi();

    %% Load model kalau ada
    if exist('chainsaw_model_v10_visual.mat','file') == 2
        load('chainsaw_model_v10_visual.mat','mu','sigma');
        fprintf('Model ditemukan -> pakai normalisasi model\n');
    else
        mu = [];
        sigma = [];
        fprintf('Model tidak ditemukan -> normalisasi dari data\n');
    end

    %% Dataset
    dataFolders = {
        fullfile(cfg.datasetRoot,'chainsaw','dekat'),                    'Chainsaw Dekat';
        fullfile(cfg.datasetRoot,'chainsaw','jauh'),                     'Chainsaw Jauh';
        fullfile(cfg.datasetRoot,'non_chainsaw_mirip','lawn_mower'),     'Lawn Mower';
        fullfile(cfg.datasetRoot,'non_chainsaw_mirip','motor_4tak_low'), 'Motor 4Tak Low';
        fullfile(cfg.datasetRoot,'non_chainsaw_mirip','motor_4tak_high'),'Motor 4Tak High';
        fullfile(cfg.datasetRoot,'non_chainsaw_mirip','motor_2tak_high'),'Motor 2Tak High';
        fullfile(cfg.datasetRoot,'non_chainsaw_mirip','mesin_listrik'),  'Mesin Listrik';
        fullfile(cfg.datasetRoot,'non_chainsaw_mirip','grass_trimmer'),  'Grass Trimmer';
        fullfile(cfg.datasetRoot,'non_chainsaw_tidak_mirip'),            'Non-CS Tidak Mirip';
    };

    fprintf('\nEkstraksi fitur untuk visualisasi...\n');

    allFeatures = [];
    allKelas = {};

    %% LOOP DATASET
    for i = 1:size(dataFolders,1)
        folderPath = dataFolders{i,1};
        namaKelas  = dataFolders{i,2};

        if ~isfolder(folderPath), continue; end

        files = [dir(fullfile(folderPath,'*.wav'));
                 dir(fullfile(folderPath,'*.mp3'))];

        for j = 1:numel(files)
            fp = fullfile(files(j).folder, files(j).name);

            try
                [y,fs] = audioread(fp);
                y = preprocessAudio(y,fs,cfg);
                if isempty(y), continue; end

                fs = cfg.targetFs;
                segSamples = round(cfg.segLen*fs);
                hopSamples = round((cfg.segLen-cfg.overlap)*fs);

                starts = 1:hopSamples:(length(y)-segSamples+1);

                for k = 1:numel(starts)
                    seg  = y(starts(k):starts(k)+segSamples-1);
                    feat = ekstrakFitur(seg,fs,cfg);

                    if ~isempty(feat)
                        allFeatures = [allFeatures; feat];
                        allKelas{end+1,1} = namaKelas; % <-- FIX penting (column cell)
                    end
                end

            catch
            end
        end
    end

    %% =========================
    %% VALIDASI (SUPER PENTING)
    %% =========================
    fprintf('Jumlah fitur : %d\n', size(allFeatures,1));
    fprintf('Jumlah label : %d\n', length(allKelas));

    if size(allFeatures,1) ~= length(allKelas)
        error('Jumlah fitur dan label TIDAK SAMA! Periksa loop segment.');
    end

    %% Normalisasi
    if isempty(mu)
        mu = mean(allFeatures,1);
        sigma = std(allFeatures,0,1);
        sigma(sigma==0)=1;
    end

    featNorm = (allFeatures - mu) ./ sigma;

    %% =========================
    %% t-SNE
    %% =========================
    fprintf('Menjalankan t-SNE...\n');

    % Ambil subset kalau terlalu besar (biar cepat & stabil)
    maxSample = 3000;
    if size(featNorm,1) > maxSample
        idx = randperm(size(featNorm,1), maxSample);
        featNorm = featNorm(idx,:);
        allKelas = allKelas(idx);
    end

    Y = tsne(featNorm, 'NumDimensions', 2, 'Perplexity', 30);

    %% =========================
    %% PLOT
    %% =========================
    figure;

    gscatter(Y(:,1), Y(:,2), allKelas);

    title('t-SNE Feature Space (Per Kelas)');
    xlabel('Dim 1'); ylabel('Dim 2');
    grid on;
    legend('Location','bestoutside');

end


%% =========================================================
%%  FUNGSI KONFIGURASI
%% =========================================================
function cfg = buatKonfigurasi()
    cfg.datasetRoot = 'dataset';
    cfg.targetFs    = 16000;   %% INMP441 via ESP32 kirim 16kHz
    cfg.segLen      = 2.0;     %% 2 detik per segmen (realtime-friendly)
    cfg.overlap     = 1.0;     %% 50% overlap
    cfg.numMFCC     = 13;      %% [Gnamele 2019]
    cfg.frameLen    = 0.025;   %% 25ms frame [Mporas 2020]
    cfg.frameHop    = 0.010;   %% 10ms hop
    cfg.minDurasi   = 0.5;
    cfg.serialPort  = 'COM3';  %% Ganti sesuai sistem
    cfg.baudRate    = 921600;
    cfg.alertCooldown = 10;
end


%% =========================================================
%%  FUNGSI PREPROCESSING AUDIO
%% =========================================================
function y = preprocessAudio(y, fs, cfg)
    if size(y,2)>1, y = mean(y,2); end
    if length(y)/fs < cfg.minDurasi, y=[]; return; end
    if fs ~= cfg.targetFs
        y = resample(y, cfg.targetFs, fs);
    end
    maxVal = max(abs(y));
    if maxVal < 1e-8, y=[]; return; end
    y = y / maxVal;
end


%% =========================================================
%%  FUNGSI EKSTRAKSI FITUR
%% =========================================================
function feat = ekstrakFitur(y, fs, cfg)
%% =========================================================
%%  EKSTRAKSI FITUR v9 - CHAINSAW ACOUSTIC FINGERPRINT ENGINE
%%
%%  FILOSOFI: Meniru cara manusia membedakan suara mesin:
%%
%%  1. CHAINSAW vs MOTOR 2-TAK:
%%     Chainsaw: rantai bergerak = AM modulation 15-80 Hz kuat,
%%     spectral roughness tinggi, harmonic tidak stabil.
%%     Motor 2-tak: lebih smooth, harmonic stabil, AM lemah.
%%
%%  2. CHAINSAW vs GRASS TRIMMER:
%%     Grass trimmer: RPM lebih tinggi, F0 lebih tinggi,
%%     tidak ada chain engagement transient.
%%     Chainsaw: transient impulsif saat rantai menyentuh kayu.
%%
%%  3. CHAINSAW vs MOTOR 4-TAK:
%%     Motor 4-tak: firing order 2x lebih lambat, harmonic genap
%%     lebih dominan. Chainsaw: harmonic ganjil kuat.
%%
%%  GRUP FITUR v9:
%%  [A] MFCC 20-koef + 8 statistik (mean,std,p10,p25,p75,p90,skew,kurt)
%%  [B] Harmonic Structure: rasio harmonic ganjil/genap, decay, centroid
%%  [C] Chain Modulation: AM 15-80 Hz, periodisitas envelope, crest
%%  [D] Spectral Roughness: irregularity, spectral flatness
%%  [E] Transient Density: crest factor, onset rate
%%  [F] Sub-band Temporal Dynamics: 8 band x 4 stat
%%  [G] LMFE 40-band x 3 stat (mean, std, gradient temporal)
%%  [H] Spectral Contrast 8-band (resolusi lebih tinggi)
%%  [I] Cepstral Peak Prominence (CPP) - kekuatan periodisitas
%%  [J] F0 Trajectory: jitter, shimmer-analog
%% =========================================================
    feat = [];
    try
        frameLen = round(cfg.frameLen * fs);
        frameHop = round(cfg.frameHop * fs);
        nFFT     = 2^nextpow2(frameLen);

        %% ============================================================
        %% [A] MFCC 20-koef + 8 statistik = 160 dim
        %% ============================================================
        nMFCC_A = 20;
        coeffs = mfcc(y, fs, ...
            'NumCoeffs',     nMFCC_A, ...
            'WindowLength',  frameLen, ...
            'OverlapLength', frameLen - frameHop, ...
            'FFTLength',     nFFT);
        if isempty(coeffs) || any(isnan(coeffs(:))), return; end

        mfccMean = mean(coeffs, 1);
        mfccStd  = std(coeffs,  0, 1);
        mfccP10  = prctile(coeffs, 10, 1);
        mfccP25  = prctile(coeffs, 25, 1);
        mfccP75  = prctile(coeffs, 75, 1);
        mfccP90  = prctile(coeffs, 90, 1);
        mfccSkew = skewness(coeffs, 1, 1);
        mfccKurt = kurtosis(coeffs, 1, 1);

        delta1 = hitungDelta(coeffs);
        delta2 = hitungDelta(delta1);
        d1Mean = mean(delta1,1); d1Std = std(delta1,0,1); d1Skew = skewness(delta1,1,1);
        d2Mean = mean(delta2,1); d2Std = std(delta2,0,1); d2Skew = skewness(delta2,1,1);

        %% ============================================================
        %% Setup frame-level analysis
        %% ============================================================
        frames   = buffer(y, frameLen, frameLen-frameHop, 'nodelay');
        win      = hamming(frameLen);
        nFrames  = size(frames, 2);
        freqBins = (0:nFFT/2) * fs / nFFT;
        nBins    = nFFT/2 + 1;

        logEn         = zeros(1, nFrames);
        specCentroid  = zeros(1, nFrames);
        specBandwidth = zeros(1, nFrames);
        specRolloff   = zeros(1, nFrames);
        specFlux      = zeros(1, nFrames);
        zcr           = zeros(1, nFrames);
        rmse          = zeros(1, nFrames);
        hnrVals       = zeros(1, nFrames);
        prevMag       = zeros(nBins, 1);
        crestFactor   = zeros(1, nFrames);
        specIrreg     = zeros(1, nFrames);
        specFlat      = zeros(1, nFrames);

        %% [G] LMFE 40-band
        nMelBand   = 40;
        melFilters = hitungMelFilterbank(nMelBand, nFFT, fs);
        lmfeFrames = zeros(nMelBand, nFrames);

        %% [H] Spectral Contrast 8-band
        nSC     = 8;
        scEdges = [0, 150, 300, 600, 1200, 2400, 4800, 7200, fs/2];
        scEdges(scEdges > fs/2) = fs/2;
        scPeak   = zeros(nSC, nFrames);
        scValley = zeros(nSC, nFrames);

        %% [B] Harmonic Power 8 harmonik
        nHarm   = 8;
        harmPow = zeros(nHarm, nFrames);

        %% [J] F0 dan amplitudo per frame
        f0Vals  = zeros(1, nFrames);
        ampVals = zeros(1, nFrames);

        %% ============================================================
        %% Main per-frame loop
        %% ============================================================
        for i = 1:nFrames
            frame   = frames(:,i) .* win;
            magFull = abs(fft(frame, nFFT));
            mag     = magFull(1:nBins);
            powSpec = mag.^2;
            magSum  = sum(mag) + eps;

            % ?? Fitur Domain Waktu ??????????????????????????????
            en = sum(frame.^2);
            logEn(i)  = log(en + eps);
            rmse(i)   = sqrt(mean(frame.^2));
            zcr(i)    = sum(abs(diff(sign(frame)))) / (2*frameLen);
            crestFactor(i) = (max(abs(frame)) + eps) / (rmse(i) + eps);

            % ?? Fitur Spektral ??????????????????????????????????
            specCentroid(i)  = sum(freqBins .* mag') / magSum;
            specBandwidth(i) = sqrt(sum(((freqBins - specCentroid(i)).^2) .* mag') / magSum);
            
            cumS    = cumsum(mag);
            rollIdx = find(cumS >= 0.85*cumS(end), 1);
            if isempty(rollIdx), rollIdx = numel(freqBins); end
            specRolloff(i) = freqBins(rollIdx);
            
            specFlux(i)    = sum((mag - prevMag).^2);
            prevMag        = mag;

            hnrVals(i) = hitungHNR(frame, fs);

            %% LMFE [G]
            melE = melFilters * powSpec;
            lmfeFrames(:,i) = log(melE + eps);

            %% Spectral Contrast [H]
            for sb = 1:nSC
                binL = max(1, round(scEdges(sb)   * nFFT/fs) + 1);
                binH = min(nBins, round(scEdges(sb+1) * nFFT/fs) + 1);
                if binH > binL
                    subMag = mag(binL:binH);
                    nN = max(1, round(0.02 * length(subMag)));
                    sortedSub = sort(subMag, 'descend');
                    scPeak(sb,i)   = mean(sortedSub(1:min(nN,end)));
                    scValley(sb,i) = mean(sortedSub(max(1,end-nN+1):end));
                end
            end

            %% Harmonic Power [B] via autocorrelation F0
            r    = xcorr(frame, 'coeff');
            r    = r(frameLen:end);
            minL = round(fs/800);
            maxL = min(round(fs/50), frameLen-1);
            if minL < maxL
                [peakV, peakIdx] = max(r(minL+1:maxL+1));
                f0Est = fs / (peakIdx + minL);
                f0Vals(i)  = f0Est;
                ampVals(i) = peakV;
                for h = 1:nHarm
                    hFreq = f0Est * h;
                    if hFreq < fs/2
                        hBin = round(hFreq * nFFT / fs) + 1;
                        hBin = max(1, min(nBins, hBin));
                        binRange = max(1,hBin-2):min(nBins,hBin+2);
                        harmPow(h,i) = max(powSpec(binRange));
                    end
                end
            end

            %% Spectral Irregularity [D]
            if nBins > 2
                midBins = 2:nBins-1;
                specIrreg(i) = sum(abs(mag(midBins) - ...
                    (mag(midBins-1) + mag(midBins+1))/2)) / (magSum + eps);
            end

            %% Spectral Flatness [D]
            geoMean = exp(mean(log(powSpec + eps)));
            ariMean = mean(powSpec) + eps;
            specFlat(i) = geoMean / ariMean;
        end

        %% ============================================================
        %% [G] Rangkum LMFE: mean, std, gradient temporal
        %% ============================================================
        lmfeMean = mean(lmfeFrames, 2)';
        lmfeStd  = std(lmfeFrames,  0, 2)';
        lmfeGrad = mean(abs(diff(lmfeFrames, 1, 2)), 2)';

        %% ============================================================
        %% [H] Rangkum Spectral Contrast
        %% ============================================================
        scContrast     = log(scPeak ./ (scValley + eps));
        scContrastMean = mean(scContrast, 2)';
        scContrastStd  = std(scContrast,  0, 2)';

        %% ============================================================
        %% [B] Harmonic Structure
        %% ============================================================
        harmMean = mean(harmPow, 2)';
        harmStd  = std(harmPow, 0, 2)';

        oddIdx  = [1,3,5,7]; evenIdx = [2,4,6,8];
        oddPow  = sum(harmMean(oddIdx));
        evenPow = sum(harmMean(evenIdx));
        harmOddEvenRatio = oddPow / (evenPow + eps);

        if all(harmMean > 0)
            p = polyfit(1:nHarm, log(harmMean+eps), 1);
            harmDecay = -p(1);
        else
            harmDecay = 0;
        end
        harmCentroid = sum((1:nHarm) .* harmMean) / (sum(harmMean) + eps);

        %% ============================================================
        %% [C] Chain Modulation Analysis
        %% ============================================================
        envelope = abs(hilbert(y));
        envNorm  = envelope / (mean(envelope) + eps);
        nFFT_env = 2048;
        [envPxx, envF] = pwelch(envNorm, round(fs/2), round(fs/4), nFFT_env, fs);

        idxChainAM = envF >= 15 & envF <= 80;
        idxMotorAM = envF >= 3  & envF <= 15;
        idxAMTot   = envF >= 1  & envF <= 150;
        totAMPow  = sum(envPxx(idxAMTot)) + eps;
        chainAMEn = sum(envPxx(idxChainAM)) / totAMPow;
        motorAMEn = sum(envPxx(idxMotorAM)) / totAMPow;
        chainMotorAMRatio = chainAMEn / (motorAMEn + eps);

        envPxxChain = envPxx;
        envPxxChain(~idxChainAM) = 0;
        [~, idxPeakChain] = max(envPxxChain);
        peakChainModFreq = envF(idxPeakChain);

        envCrest = max(envNorm) / (mean(envNorm) + eps);

        minLagEnv = round(fs/80); maxLagEnv = round(fs/15);
        [rEnv, lagEnv] = xcorr(envNorm - mean(envNorm), round(fs*0.5), 'coeff');
        rEnv = rEnv(lagEnv >= 0);
        if minLagEnv < maxLagEnv && maxLagEnv < length(rEnv)
            envPeriodicity = max(rEnv(minLagEnv+1:maxLagEnv+1));
        else
            envPeriodicity = 0;
        end

        envelopeVar  = var(envelope);
        envelopeSkew = skewness(envelope);
        envelopeKurt = kurtosis(envelope);

        %% ============================================================
        %% [D] Spectral Roughness
        %% ============================================================
        specIrregMean = mean(specIrreg); specIrregStd = std(specIrreg);
        specFlatMean  = mean(specFlat);  specFlatStd  = std(specFlat);

        %% ============================================================
        %% [E] Transient / Onset
        %% ============================================================
        crestMean = mean(crestFactor);
        crestStd  = std(crestFactor);
        crestMax  = max(crestFactor);
        fluxThresh   = mean(specFlux) + std(specFlux);
        onsetDensity = sum(specFlux > fluxThresh) / nFrames;

        %% ============================================================
        %% [F] Sub-band Temporal Dynamics
        %% ============================================================
        [pxx, fPxx] = pwelch(y, frameLen, frameLen-frameHop, nFFT, fs);
        subBandEdges = [0, 150, 300, 600, 1200, 2400, 4000, 6000, fs/2];
        subBandEdges(subBandEdges > fs/2) = fs/2;
        nSB = 8;
        sbEnergy   = zeros(1, nSB);
        totPow     = sum(pxx) + eps;
        for sb = 1:nSB
            idxSB = fPxx >= subBandEdges(sb) & fPxx < subBandEdges(sb+1);
            sbEnergy(sb) = sum(pxx(idxSB)) / totPow;
        end

        melPerGroup = floor(nMelBand / nSB);
        sbTempMean = zeros(1,nSB); sbTempStd = zeros(1,nSB);
        sbTempSkew = zeros(1,nSB); sbTempKurt = zeros(1,nSB);
        for sb = 1:nSB
            idxS = (sb-1)*melPerGroup + 1;
            idxE = min(sb*melPerGroup, nMelBand);
            bandEnergy = mean(lmfeFrames(idxS:idxE, :), 1);
            sbTempMean(sb) = mean(bandEnergy);
            sbTempStd(sb)  = std(bandEnergy);
            sbTempSkew(sb) = skewness(bandEnergy);
            sbTempKurt(sb) = kurtosis(bandEnergy);
        end

        %% ============================================================
        %% [I] Cepstral Peak Prominence (CPP)
        %% ============================================================
        cepstrum = real(ifft(log(abs(fft(y, nFFT)).^2 + eps)));
        cepstrum = cepstrum(1:nFFT/2);
        minQ = round(fs/800);
        maxQ = min(round(fs/50), length(cepstrum)-1);
        if minQ < maxQ
            [cppPeakVal, cppPeakIdx] = max(cepstrum(minQ+1:maxQ+1));
            cppPeakIdx = cppPeakIdx + minQ;
            qRange  = (minQ:maxQ)';
            p_cpp   = polyfit(qRange, cepstrum(minQ+1:maxQ+1), 1);
            baseline = polyval(p_cpp, cppPeakIdx);
            cpp = cppPeakVal - baseline;
        else
            cpp = 0;
        end

        %% ============================================================
        %% [J] F0 Trajectory: Jitter & Shimmer
        %% ============================================================
        validF0  = f0Vals(f0Vals > 0);
        validAmp = ampVals(ampVals > 0);
        if numel(validF0) > 2
            f0Mean    = mean(validF0);
            f0Std     = std(validF0);
            f0Var     = var(validF0);
            f0Jitter  = mean(abs(diff(validF0))) / (f0Mean + eps);
            f0RJitter = std(validF0) / (f0Mean + eps);
        else
            f0Mean = 0; f0Std = 0; f0Var = 0; f0Jitter = 0; f0RJitter = 0;
        end
        if numel(validAmp) > 2
            shimmer = mean(abs(diff(validAmp))) / (mean(validAmp) + eps);
        else
            shimmer = 0;
        end

        %% Fitur global standar
        [~, idxMax]  = max(pxx);
        dominantFreq = fPxx(idxMax);
        zcrThresh    = median(zcr);
        voicingProb  = sum(zcr < zcrThresh) / nFrames;
        specFluxVar  = var(specFlux);
        zcrVar       = var(zcr);
        rmseVar      = var(rmse);

        %% ============================================================
        %% GABUNGKAN SEMUA FITUR v9
        %% ============================================================
        feat = [ ...
            % [A] MFCC + Delta + Delta-Delta (280 dim)
            mfccMean, mfccStd, mfccP10, mfccP25, mfccP75, mfccP90, mfccSkew, mfccKurt, ...
            d1Mean, d1Std, d1Skew, ...
            d2Mean, d2Std, d2Skew, ...
            % [B] Harmonik + HNR (23 dim)
            harmMean, harmStd, ...
            harmOddEvenRatio, harmDecay, harmCentroid, ...
            mean(hnrVals), std(hnrVals), skewness(hnrVals), kurtosis(hnrVals), ...
            % [C] Modulasi Amplitudo (9 dim)
            chainAMEn, motorAMEn, chainMotorAMRatio, ...
            peakChainModFreq, envCrest, envPeriodicity, ...
            envelopeVar, envelopeSkew, envelopeKurt, ...
            % [D] Fitur Tambahan (8 dim)
            specIrregMean, specIrregStd, specFlatMean, specFlatStd, ...
            crestMean, crestStd, crestMax, onsetDensity, ...
            % [E] Sub-band Energy + Temporal (40 dim)
            sbEnergy, sbTempMean, sbTempStd, sbTempSkew, sbTempKurt, ...
            % [F] LMFE (3 statistik � 40 filter Mel): 120 dim
            lmfeMean, lmfeStd, lmfeGrad, ...
            % [G] Spectral Contrast (2 statistik � 8 pita): 16 dim
            scContrastMean, scContrastStd, ...
            % [H] Fitur Tambahan (CPP, F0 & Shimmer, Dominant Freq & Voicing): 10 dim
            cpp, ...
            f0Mean, f0Std, f0Var, f0Jitter, f0RJitter, shimmer, ...
            dominantFreq, voicingProb, ...
            % [K] Fitur Spektral Dasar (17 dim)
            mean(logEn), std(logEn), ...
            mean(specCentroid), std(specCentroid), ...
            mean(specBandwidth), std(specBandwidth), ...
            mean(specRolloff), std(specRolloff), ...
            mean(specFlux), std(specFlux), specFluxVar, ...
            mean(zcr), std(zcr), zcrVar, ...
            mean(rmse), std(rmse), rmseVar ...
        ];

        if any(isnan(feat)) || any(isinf(feat))
            feat = [];
        end
    catch
        feat = [];
    end
end


%% =========================================================
%%  [v10] FUNGSI AUGMENTASI SEGMEN
%%  ----------------------------------------------------------
%%  Memperbanyak 1 segmen audio menjadi 6 versi berbeda
%%  untuk meningkatkan keragaman data training tanpa rekam ulang.
%%
%%  Input:
%%    seg - segmen audio asli [Nx1 float, sudah dinormalisasi -1..1]
%%    fs  - sample rate (Hz)
%%  Output:
%%    segList - cell array 1x6 berisi versi augmented
%%
%%  6 Teknik Augmentasi:
%%  [1] Asli          - tidak diubah sama sekali
%%  [2] White noise   - tambah noise Gaussian ringan ~30dB SNR
%%                      Simulasi: rekaman di kondisi sedikit berisik
%%  [3] Time stretch+ - percepat 10% via resample (9/10)
%%                      Simulasi: mesin RPM sedikit lebih tinggi
%%  [4] Time stretch- - perlambat 10% via resample (11/10)
%%                      Simulasi: mesin RPM sedikit lebih rendah
%%  [5] Pitch shift+2 - naikkan pitch 2 semitone (faktor 1.1225)
%%                      Simulasi: chainsaw berbeda merek/ukuran
%%  [6] Pitch shift-2 - turunkan pitch 2 semitone (faktor 0.8909)
%%                      Simulasi: chainsaw berbeda merek/ukuran
%%
%%  Catatan implementasi:
%%  - Menggunakan resample() MATLAB standar (tersedia semua versi)
%%  - Semua hasil dipastikan panjang = panjang asli (crop/zero-pad)
%%  - Jika augmentasi gagal karena error, fallback ke segmen asli
%%  - Normalisasi ulang setelah setiap augmentasi
%% =========================================================
function segList = augmentasiSegmen(seg, fs)
    N       = length(seg);
    segList = cell(1, 6);

    %% ---- [1] Original ----
    segList{1} = seg;

    %% ---- [2] White noise ringan ----
    try
        noiseLevel = 0.005 * std(seg);        %% ~30dB SNR
        segNoise   = seg + noiseLevel * randn(N, 1);
        segNoise   = max(min(segNoise, 1.0), -1.0);  %% clip
        segList{2} = segNoise;
    catch
        segList{2} = seg;  %% fallback
    end

    %% ---- [3] Time stretch +10% (lebih cepat) ----
    %% resample(x, P, Q) mengubah panjang jadi N*P/Q
    %% P=9, Q=10 -> panjang baru = 90% dari N
    %% Ambil 90% pertama, pad nol sisanya agar tetap panjang N
    try
        yFast = resample(double(seg(:)), 9, 10);
        yFast = yFast(:);
        nF    = length(yFast);
        if nF >= N
            out = yFast(1:N);
        else
            out = [yFast; zeros(N - nF, 1)];
        end
        mxF = max(abs(out));
        if mxF > 1e-8, out = out / mxF; end
        segList{3} = out;
    catch
        segList{3} = seg;
    end

    %% ---- [4] Time stretch -10% (lebih lambat) ----
    %% P=11, Q=10 -> panjang baru = 110% dari N
    %% Ambil N sampel pertama (crop)
    try
        ySlow = resample(double(seg(:)), 11, 10);
        ySlow = ySlow(:);
        nS    = length(ySlow);
        if nS >= N
            out = ySlow(1:N);
        else
            out = [ySlow; zeros(N - nS, 1)];
        end
        mxS = max(abs(out));
        if mxS > 1e-8, out = out / mxS; end
        segList{4} = out;
    catch
        segList{4} = seg;
    end

    %% ---- [5] Pitch shift +2 semitone ----
    %% Faktor frekuensi: 2^(2/12) = 1.1225
    %% Cara: resample ke fs*1.1225 lalu kembalikan ke panjang N
    try
        factorUp = 2^(2/12);              %% = 1.1225
        fsUp     = round(fs * factorUp);
        yUp      = resample(double(seg(:)), fsUp, fs);
        yUp      = yUp(:);
        nU       = length(yUp);
        if nU >= N
            out = yUp(1:N);
        else
            out = [yUp; zeros(N - nU, 1)];
        end
        mxU = max(abs(out));
        if mxU > 1e-8, out = out / mxU; end
        segList{5} = out;
    catch
        segList{5} = seg;
    end

    %% ---- [6] Pitch shift -2 semitone ----
    %% Faktor frekuensi: 2^(-2/12) = 0.8909
    try
        factorDn = 2^(-2/12);             %% = 0.8909
        fsDn     = max(1, round(fs * factorDn));
        yDn      = resample(double(seg(:)), fsDn, fs);
        yDn      = yDn(:);
        nD       = length(yDn);
        if nD >= N
            out = yDn(1:N);
        else
            out = [yDn; zeros(N - nD, 1)];
        end
        mxD = max(abs(out));
        if mxD > 1e-8, out = out / mxD; end
        segList{6} = out;
    catch
        segList{6} = seg;
    end
end


%% =========================================================
%%  [F1] HELPER: Hitung Mel Filterbank
%% =========================================================
function filters = hitungMelFilterbank(nBand, nFFT, fs)
    %% Konversi Hz ke Mel dan balik
    hz2mel  = @(hz) 2595 * log10(1 + hz/700);
    mel2hz  = @(mel) 700 * (10.^(mel/2595) - 1);

    melLow  = hz2mel(0);
    melHigh = hz2mel(fs/2);
    melPts  = linspace(melLow, melHigh, nBand+2);
    hzPts   = mel2hz(melPts);
    binPts  = floor((nFFT+1) * hzPts / fs);

    filters = zeros(nBand, nFFT/2+1);
    for m = 1:nBand
        fStart = binPts(m);
        fPeak  = binPts(m+1);
        fEnd   = binPts(m+2);
        for k = fStart:fPeak
            if fPeak > fStart
                filters(m, k+1) = (k - fStart) / (fPeak - fStart);
            end
        end
        for k = fPeak:fEnd
            if fEnd > fPeak
                filters(m, k+1) = (fEnd - k) / (fEnd - fPeak);
            end
        end
    end
end



function d = hitungDelta(C)
    d = zeros(size(C));
    N = size(C,1);
    for t=1:N
        tm2=max(1,t-2); tm1=max(1,t-1);
        tp1=min(N,t+1); tp2=min(N,t+2);
        d(t,:) = (-2*C(tm2,:)-C(tm1,:)+C(tp1,:)+2*C(tp2,:))/10;
    end
end


%% =========================================================
%%  FIX BUG #2: Range HNR dipersempit ke 80-400 Hz
%%  Alasan: Motor 2-tak high RPM punya harmonic kuat di
%%  50-500 Hz yang identik chainsaw jika range terlalu lebar.
%%  Range 80-400 Hz lebih selektif untuk chainsaw.
%%  Tambahan: kembalikan juga variance HNR antar frame
%%  sebagai fitur tambahan (chainsaw lebih tidak stabil).
%% =========================================================
function hnr = hitungHNR(frame, fs)
    try
        N    = length(frame);
        r    = xcorr(frame, 'coeff');
        r    = r(N:end);
        %% FIX: range 80-400 Hz (bukan 50-500 Hz)
        minLag = round(fs/400);   %% was fs/500 = 32 samp -> now fs/400 = 40 samp
        maxLag = min(round(fs/80), N-1);  %% was fs/50 = 320 -> now fs/80 = 200
        if minLag >= maxLag, hnr=0; return; end
        [peakVal,~] = max(r(minLag+1:maxLag+1));
        peakVal = min(max(peakVal,-0.9999),0.9999);
        hnr = 10*log10(peakVal^2/(1-peakVal^2+eps));
    catch
        hnr = 0;
    end
end


function gambarBoxPlot(ax, data, xPos)
    if numel(data)<2, return; end
    q      = quantile(data,[0.25 0.50 0.75]);
    iqrVal = q(3)-q(1);
    wLow   = max(min(data), q(1)-1.5*iqrVal);
    wHigh  = min(max(data), q(3)+1.5*iqrVal);
    w = 0.30;
    rectangle('Parent',ax,'Position',[xPos-w/2,q(1),w,max(iqrVal,0.01)], ...
        'FaceColor',[0.6 0.8 1 0.7],'EdgeColor',[0.2 0.4 0.8],'LineWidth',1.2);
    plot(ax,[xPos-w/2 xPos+w/2],[q(2) q(2)],'r-','LineWidth',2);
    plot(ax,[xPos xPos],[wLow q(1)],'k-');
    plot(ax,[xPos xPos],[q(3) wHigh],'k-');
    plot(ax,[xPos-w/4 xPos+w/4],[wLow wLow],'k-');
    plot(ax,[xPos-w/4 xPos+w/4],[wHigh wHigh],'k-');
    out = data(data<wLow|data>wHigh);
    if ~isempty(out)
        plot(ax,repmat(xPos,1,numel(out)),out,'k.','MarkerSize',6);
    end
end




