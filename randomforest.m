%% Clean start
close all; clear; clc;
rng(42); % Ensure reproducibility

%% Load data
trainData = readtable('train.csv'); 
testData  = readtable('test.csv');

%% Statistical Analysis of the Dataset
summary(trainData);
% Overview of dataset
summary(trainData)

% Count survivors
tabulate(trainData.Survived)

% Average age by survival
grpstats(trainData, 'Survived', {'mean'}, 'DataVars', 'Age')

% Gender vs survival
tabulate(trainData.Sex)
crosstab(trainData.Sex, trainData.Survived)

% Class distribution
tabulate(trainData.Pclass)

% Visuals (optional)
figure;
histogram(trainData.Age);
title('Age Distribution');

figure;
bar(crosstab(trainData.Sex, trainData.Survived));
title('Survival by Gender');
xlabel('Gender'); ylabel('Count');
legend('Did Not Survive', 'Survived');

%% Data Cleaning - TRAIN DATA

% Fill Age missing values using median per Pclass & Sex
for p = 1:3
    for s = ["male","female"]
        mask = (trainData.Pclass == p) & strcmp(trainData.Sex,s);
        medAge = median(trainData.Age(mask),'omitnan');
        trainData.Age(mask & isnan(trainData.Age)) = medAge;
    end
end

% Convert Embarked to categorical
trainData.Embarked = categorical(trainData.Embarked);

% Fill missing Embarked with its mode
modeEmbarked = mode(trainData.Embarked);
trainData.Embarked = fillmissing(trainData.Embarked,'constant',modeEmbarked);

% Drop Cabin and Ticket
trainData.Cabin = [];
trainData.Ticket = [];

% Verify missing values
missingCounts = sum(ismissing(trainData));
T = table(trainData.Properties.VariableNames', missingCounts', ...
          'VariableNames', {'Column', 'MissingCount'});
disp(T)

%% Feature Engineering - TRAIN DATA

trainData.FamilySize = trainData.SibSp + trainData.Parch + 1;
trainData.IsAlone = trainData.FamilySize == 1;
trainData.Title = extractBetween(trainData.Name, ", ", ".");
trainData.Title = strtrim(trainData.Title);
trainData.Title = categorical(trainData.Title);

% Encode categoricals
trainData.Sex = categorical(trainData.Sex);
trainData.Embarked = categorical(trainData.Embarked);

%% Define Features (X) and Target (y)
X = removevars(trainData, {'PassengerId','Name','Survived'});
y = trainData.Survived;

%% Data Cleaning - TEST DATA

% Fill missing Age in test set using the same logic
for p = 1:3
    for s = ["male","female"]
        mask = (testData.Pclass == p) & strcmp(testData.Sex,s);
        medAge = median(testData.Age(mask),'omitnan');
        testData.Age(mask & isnan(testData.Age)) = medAge;
    end
end

% Convert Embarked to categorical
testData.Embarked = categorical(testData.Embarked);

% Fill missing Embarked with mode from TRAIN (important!)
testData.Embarked = fillmissing(testData.Embarked, 'constant', modeEmbarked);

% Drop Cabin and Ticket
testData.Cabin = [];
testData.Ticket = [];

%% Feature Engineering - TEST DATA
testData.FamilySize = testData.SibSp + testData.Parch + 1;
testData.IsAlone = testData.FamilySize == 1;
testData.Title = extractBetween(testData.Name, ", ", ".");
testData.Title = strtrim(testData.Title);
testData.Title = categorical(testData.Title);

% Encode categoricals
testData.Sex = categorical(testData.Sex);
testData.Embarked = categorical(testData.Embarked);

%% Prepare test features
Xtest = removevars(testData, {'PassengerId','Name'});

disp('Training and test data cleaned and preprocessed successfully!')

%% Train-Validation Split

cv = cvpartition(height(trainData), 'HoldOut', 0.2);
idxTrain = training(cv);
idxVal = test(cv);

Xtrain = X(idxTrain, :);
ytrain = y(idxTrain, :);

Xval = X(idxVal, :);
yval = y(idxVal, :);

%% Model Training - Random Forest
model = fitcensemble(Xtrain, ytrain, ...
    'Method', 'Bag', ...
    'NumLearningCycles', 100, ...
    'Learners', templateTree('MaxNumSplits', 20));

%% Model Evaluation

yPred = predict(model, Xval);
acc = mean(yPred == yval);
fprintf('Validation Accuracy: %.2f%%\n', acc * 100);

% Confusion matrix
figure;
confusionchart(yval, yPred);
title('Confusion Matrix - Validation Set');

%% Predict on Test Set

yTestPred = predict(model, Xtest);

submission = table(testData.PassengerId, yTestPred, ...
    'VariableNames', {'PassengerId', 'Survived'});

writetable(submission, 'submission.csv');
disp('submission.csv created successfully!');
