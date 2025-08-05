clear;
clc;
%% Connect the EEG device
delete(instrfindall);
s = serial('COM3', 'BaudRate', 115200);
fopen(s);

%% Start collecting EEG signals
disp('We are collecting EEG signals...');
% Initialize Psychtoolbox
PsychDefaultSetup(2);
Screen('Preference', 'SkipSyncTests', 1);
% Get screen number
screenNumber = max(Screen('Screens'));
% Get screen size
screenRect = Screen('Rect', screenNumber);
% Open a blank window
[window, windowRect] = Screen('OpenWindow', screenNumber, [0 0 0], screenRect);
% Set text size
textSize = 40;
% Set initial screen text color to white
textColor = [255 255 255];
% Set emotion induction screen text color to black
textColor1 = [0 0 0];
% Display text: Prepare to view pictures of car interiors
Screen('TextSize', window, textSize); % Set text size
DrawFormattedText(window, 'Prepare to view pictures of car interiors.', 'center', 'center', textColor);
Screen('Flip', window); % Update screen
WaitSecs(1); % Wait for 1 second
duration = 120; % Total duration, designed as rounds*6
delay = 1;
% Initialize timer
timer = 0;

% Calculate rounds, with 120s as the duration for each round
durationPerRound = duration;

while timer < duration  
    % Start a new trial every 120 seconds
    % At the start of each round, display a cross
    if mod(timer, durationPerRound) == 0
        Screen('TextSize', window, textSize); % Set text size
        DrawFormattedText(window, 'Prepare to view pictures of car interiors.', 'center', 'center', textColor);
        Screen('Flip', window);
        beep;
    end
    
    %% Display stimulus screen at 5 seconds
    if mod(timer, durationPerRound) == 5
        % Select a different image file for each round. After each round, change the image (expected 16 images, 10 for cross-validation and training; 6 for testing)
        imageFile = sprintf('D:\\2025\\experiment\\car1\\car1.jpg');
        % Read and draw the image
        imageData = imread(imageFile); % Read image data
        texture = Screen('MakeTexture', window, imageData); % Create texture
        % Get image size
        imageSize = size(imageData);
        imageWidth = imageSize(2);
        imageHeight = imageSize(1);
        % Calculate image scaling ratio
        scale = min(screenRect(3) / imageWidth, screenRect(4) / imageHeight);
        scaledWidth = imageWidth * scale;
        scaledHeight = imageHeight * scale;
        % Set image position
        destRect = [0 0 scaledWidth scaledHeight]; % Target rectangle for image
        destRect = CenterRectOnPoint(destRect, screenRect(3) / 2, screenRect(4) / 2); % Center image
        % Draw the image
        Screen('DrawTexture', window, texture, [], destRect); % Draw the image
        % Label for data extraction (optional)
        fwrite(s,[1 225 1 0 1]);
        Screen('Flip', window); % Update screen
    end

    %% Display three colored rectangles (red, white, blue) at 50 seconds
    if mod(timer, durationPerRound) == 50
        % Set rectangle size and spacing
        rectWidth = 450; % Rectangle width
        rectHeight = 300; % Rectangle height
        gap = 30; % Rectangle spacing

        % Calculate rectangle center positions
        leftRectCenterX = screenRect(3) / 4 - rectWidth / 2 - gap / 2;
        centerRectCenterX = screenRect(3) / 2;
        rightRectCenterX = 3 * screenRect(3) / 4 + rectWidth / 2 + gap / 2;
        rectCenterY = screenRect(4) / 2;

        % Draw red rectangle
        rectRed = CenterRectOnPoint([0 0 rectWidth rectHeight], leftRectCenterX, rectCenterY);
        Screen('FillRect', window, [255 0 0], rectRed);
        % Draw "Good" text inside red rectangle
        DrawFormattedText(window, 'Good', 'center', 'center', textColor1, [], [], [], [], [], rectRed);
    
        % Draw white rectangle
        rectWhite = CenterRectOnPoint([0 0 rectWidth rectHeight], centerRectCenterX, rectCenterY);
        Screen('FillRect', window, [255 255 255], rectWhite);
        % Draw "Normal" text inside white rectangle
        DrawFormattedText(window, 'Normal', 'center', 'center', textColor1, [], [], [], [], [], rectWhite);

        % Draw blue rectangle
        rectBlue = CenterRectOnPoint([0 0 rectWidth rectHeight], rightRectCenterX, rectCenterY);
        Screen('FillRect', window, [0 0 255], rectBlue);
        % Draw "Bad" text inside blue rectangle
        DrawFormattedText(window, 'Bad', 'center', 'center', textColor1, [], [], [], [], [], rectBlue);

        % Display text
        DrawFormattedText(window, 'Please make your judgment:', 'center', 100, textColor);
        fwrite(s,[1 225 1 0 4]);
        % Update screen
        Screen('Flip', window);
        beep;
    end

    %% Display questionnaire at 58 seconds
    if mod(timer, durationPerRound) == 58
        % Set text size and color
        Screen('TextSize', window, 24);
        textColor = [255 255 255];

        % Load image, same image used in emotion induction phase
        image = imread(imageFile);
        texture = Screen('MakeTexture', window, image);
        % Define questions and options
        questions = {
            '1. Are the gearshift lever layout and design coordinated and aesthetically pleasing??',
            '2. Is the design of the steering wheel beautiful?',
            '3. Do the control buttons on the door (such as window switches) fit in with the overall design?',
            '4. Is the seat comfortable based on appearance alone?',
            '5. What do you think of the design of the center console?',
            '6. Is the design of the other visible areas of the car consistent?',
            '7. What do you think of the overall interior design of the car?'
        };
        options = {
            {'1. Very dissatisfied', '2. Dissatisfied', '3. Neutral', '4. Satisfied', '5. Very satisfied'},
            {'1. Very dissatisfied', '2. Dissatisfied', '3. Neutral', '4. Satisfied', '5. Very satisfied'},
            {'1. Very dissatisfied', '2. Dissatisfied', '3. Neutral', '4. Satisfied', '5. Very satisfied'},
            {'1. Very dissatisfied', '2. Dissatisfied', '3. Neutral', '4. Satisfied', '5. Very satisfied'},
            {'1. Very dissatisfied', '2. Dissatisfied', '3. Neutral', '4. Satisfied', '5. Very satisfied'},
            {'1. Very dissatisfied', '2. Dissatisfied', '3. Neutral', '4. Satisfied', '5. Very satisfied'},
            {'1. Very dissatisfied', '2. Dissatisfied', '3. Neutral', '4. Satisfied', '5. Very satisfied'}
        };
        keys = {'1', '2', '3', '4', '5'};
        
        % Image position
        imageRect = [0, 0, 750, 480]; % Image size
        imagePos = CenterRectOnPointd(imageRect, windowRect(3) / 4, windowRect(4) / 2);

        % Open file to save results, remember to modify each time to avoid confusion
        fileID = fopen('D:\\2025\\experiment\\scheme1\\survey_1.txt', 'a'); % Open file in append mode

        % Loop through each question and record response
        for i = 1:length(questions)
            question = questions{i};
            optionList = options{i};
            
            % Display image
            Screen('DrawTexture', window, texture, [], imagePos);
            
            % Display prompt text
            promptText = 'Please fill out the questionnaire';
            promptPos = [windowRect(3) / 2, windowRect(4) / 8];
            DrawFormattedText(window, promptText, promptPos(1), promptPos(2), textColor);

            % Display question and options
            questionPos = [windowRect(3) / 2, windowRect(4) / 4];
            DrawFormattedText(window, question, questionPos(1), questionPos(2), textColor);
            for j = 1:length(optionList)
                optionPos = [windowRect(3) / 2, windowRect(4) / 4 + j * 60];
                DrawFormattedText(window, optionList{j}, optionPos(1), optionPos(2), textColor);
            end
            Screen('Flip', window);
    
            % Wait for response
            response = '';
            while isempty(response)
                [keyIsDown, ~, keyCode] = KbCheck;
                if keyIsDown
                    for k = 1:length(keys)
                        if keyCode(KbName(keys{k}))
                            response = optionList{k};
                            break;
                        end
                    end
                end
            end
    
            % Display response
            Screen('DrawTexture', window, texture, [], imagePos);
            DrawFormattedText(window, ['Your response: ', response], questionPos(1), questionPos(2), textColor);
            Screen('Flip', window);
            WaitSecs(2);
    
            % Save results to file
            fprintf(fileID, 'Question: %s\n', question);
            fprintf(fileID, 'Response: %s\n\n', response);
        end
        % Close file
        fclose(fileID);

        % Close Psychtoolbox
        Screen('CloseAll');
        disp('Survey completed, program ends.');
        return;
    end

    WaitSecs(delay);
    % Check for escape key event to quit
    escapeKey = KbName('ESCAPE');
    [~, ~, keyCode] = KbCheck;
    if keyCode(escapeKey)
        Screen('CloseAll');
        disp('Program terminated by user.');
        return;
    end
    % Increment timer
    timer = timer + 1;
end
fclose(dataClient);
fclose(s);
