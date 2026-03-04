%% Track red dots in a video (auto-detect + optional manual edits)
% Workflow:
%  1) Load video
%  2) On the first frame, user draws a rectangle around the object area
%  3) Crop vertically around that rectangle (with margins)
%  4) Auto-detect candidate dots in cropped region
%  5) User can add points (left click), finish with Enter / right click
%  6) User can remove undesired points by clicking them, finish with Enter
%  7) Track each dot across frames by local max "redness" in a small window
%  8) Save tracked positions to a .mat file

clear; close all; clc;

%% Parameters
startIdx = 575;
endIdx = 575;

% Crop margins around the rectangle (in pixels)
delta_inf = 50;   % extra pixels below the rectangle bottom
delta_sup = 200;  % extra pixels above the rectangle top

% Detection parameters (first frame)
threshold_detect   = 0.3;   % binarization threshold on processed image
minArea_keep   = 20;    % remove very small objects
minArea_smallObj  = 450;   % "small object" threshold (used to separate small vs big blobs)

% Tracking parameters
half_window   = 10;   % half size of search window around previous position (pixels)
max_drift_from_init = 100;  % max distance from initial position allowed (pixels)

%% -------- Main loop over videos --------
for idx = startIdx:endIdx

    % Build file name: C0575.avi, etc.
    vidName = sprintf('C%04d', idx);
    fname   = fullfile('compressed', [vidName, '.avi']);

    if ~exist(fname, 'file')
        warning("File '%s' not found. Skipping.", fname);
        continue;
    end

    % Open video
    v = VideoReader(fname);
    try
        numFrames = v.NumFrames;
    catch
        numFrames = floor(v.Duration * v.FrameRate);
    end

    %% 1) Read first frame and user selects a rectangle 
    % Use readFrame (not readFrame after doing readFrame twice by accident)
    firstFrame = im2double(readFrame(v));

    hFig = figure('Name', sprintf('Automatic point detection - %s', fname));
    imshow(firstFrame);
    title('Draw a rectangle around the object');

    uiwait(helpdlg('Draw a rectangle around the object, then double-click to validate.'));
    rect = getrect; % [xmin ymin width height]
    close(hFig);

    % Rectangle coordinates (image coordinates):
    % rect(2) is y of top-left corner. rect(4) is height.
    y_top    = round(rect(2));
    y_bottom = round(rect(2) + rect(4));

    % Define vertical crop bounds with margins, clamped to image size.
    y_min_crop = max(1, y_top    - delta_sup);
    y_max_crop = min(size(firstFrame,1), y_bottom + delta_inf);

    % The cropped image height (used later in tracking bounds)
    cropH = y_max_crop - y_min_crop + 1;

    pause(0.2);

    %% 2) Auto-detect dots in cropped region (first frame)
    % dots are red → use (R - G) as a simple red emphasis for detection.
    detectImg = medfilt2( firstFrame(y_min_crop:y_max_crop,:,1) - firstFrame(y_min_crop:y_max_crop,:,2) );
    detectImg = imadjust(detectImg);

    binaryImg = imbinarize(detectImg, threshold_detect);

    % Remove small objects (keep only blobs larger than minArea_smallObj)
    largeOnly = bwareaopen(binaryImg, minArea_smallObj);

    % Remove tiny noise from remaining candidates
    niceImg = bwareaopen(smallCandidate, minArea_keep);

    % Compute centroids of remaining blobs
    stats = regionprops(niceImg, 'Centroid');
    if isempty(stats)
        centroids_auto = zeros(0,2);
    else
        centroids_all = cat(1, stats.Centroid);

        % Filter by y position (your original logic, kept but clarified):
        % centroids are in cropped coordinates (y=1 corresponds to y_min_crop in full frame)
        keepY = centroids_all(:,2) > delta_sup;  % below the top margin zone
        centroids_f1 = centroids_all(keepY,:);

        keepY2 = centroids_f1(:,2) > 10;         % redundant safety, kept
        centroids_f2 = centroids_f1(keepY2,:);

        % Keep unique-ish x positions (reduce duplicates)
        % uniquetol returns unique values; you used it inside find().
        % We'll keep the indices of unique x values by tolerance.
        [~, uniqueIdx] = uniquetol(centroids_f2(:,1), 50, 'DataScale', 1);
        centroids_auto = centroids_f2(sort(uniqueIdx), :);
    end

    %% 3) Let user add points manually
    figure('Name', 'Auto detected points (add more if needed)');
    imshow(firstFrame(y_min_crop:y_max_crop,:,:)); hold on;
    plot(centroids_auto(:,1), centroids_auto(:,2), 'r*');
    title(['Auto-detected points. Press Enter if OK, ' ...
           'or click to add points (right-click to finish).']);

    % getpts:
    % - Left click adds points
    % - Right click or Enter finishes
    try
        [x_manual, y_manual] = getpts;
        centroids_manual = [x_manual, y_manual];
    catch
        centroids_manual = zeros(0,2);
    end

    centroids_init = [centroids_auto; centroids_manual];

    %% 4) Let user remove points by clicking
    figure('Name', 'Remove undesired points');
    imshow(firstFrame(y_min_crop:y_max_crop,:,:)); hold on;
    plot(centroids_init(:,1), centroids_init(:,2), 'g*', 'MarkerSize', 8);
    title('Click points to remove. Press Enter to validate.');

    [x_deselect, y_deselect] = getpts;

    points_to_remove = [];
    for k = 1:numel(x_deselect)
        % Find closest point to click
        d = hypot(centroids_init(:,1) - x_deselect(k), ...
                  centroids_init(:,2) - y_deselect(k));
        [~, idx_min] = min(d);
        points_to_remove(end+1) = idx_min; %#ok<SAGROW>
    end

    % Remove duplicates in removal list, then remove those points
    points_to_remove = unique(points_to_remove);
    centroids_init(points_to_remove, :) = [];

    close;

    %% 5) Track points across frames
    N = size(centroids_init, 1);
    position = nan(numFrames, N, 2);
    position(1,:,:) = centroids_init;

    % Rewind video: create a new VideoReader to restart from frame 1
    v = VideoReader(fname);
    frameIndex = 0;

    figure('Name', sprintf('Tracking - %s', vidName));

    while hasFrame(v)
        frame = readFrame(v);
        frameIndex = frameIndex + 1;

        % Crop and compute a "redness" score:
        % redness = R - 0.6G - 0.6B (your original heuristic)
        r = double(frame(y_min_crop:y_max_crop,:,1));
        g = double(frame(y_min_crop:y_max_crop,:,2));
        b = double(frame(y_min_crop:y_max_crop,:,3));
        redness = r - 0.6*g - 0.6*b;

        for i = 1:N

            % Get previous position (or initial if first frame)
            if frameIndex == 1 || isnan(position(frameIndex-1,i,1))
                x_prev = centroids_init(i,1);
                y_prev = centroids_init(i,2);
            else
                x_prev = position(frameIndex-1,i,1);
                y_prev = position(frameIndex-1,i,2);
            end

            % Define a search window around previous location
            xmin = round(max(x_prev - half_window, 1));
            xmax = round(min(x_prev + half_window, size(frame,2)));

            ymin_win = round(max(y_prev - half_window, 1));
            ymax_win = round(min(y_prev + half_window, cropH));

            window = redness(ymin_win:ymax_win, xmin:xmax);

            % Find the max redness pixel in that window
            [yy, xx] = find(window == max(window(:)), 1, 'first');

            x_new = xmin + xx - 1;
            y_new = ymin_win + yy - 1;

            % Validate drift relative to the INITIAL position (your original rule)
            x_ref = position(1, i, 1);
            y_ref = position(1, i, 2);
            dist_from_init = hypot(x_new - x_ref, y_new - y_ref);

            if dist_from_init < max_drift_from_init
                position(frameIndex, i, :) = [x_new, y_new];
            else
                position(frameIndex, i, :) = [NaN, NaN];
            end
        end

        % Display tracking result on current frame crop
        imshow(frame(y_min_crop:y_max_crop,:,:)); hold on;
        for i = 1:N
            plot(position(frameIndex,i,1), position(frameIndex,i,2), 'b*');
        end
        hold off;
        title(sprintf('Frame %d', frameIndex));
        drawnow;
    end

    %% 6) Save result
    save([vidName, '_positions.mat'], 'position');
    fprintf("Done %s\n", vidName);

end