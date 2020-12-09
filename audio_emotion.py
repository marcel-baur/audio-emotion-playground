#
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.regularizers import l2
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import pickle
from sklearn import tree

from pathlib import Path

from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.applications.vgg16 import VGG16


def generate_plots(path, name, folder="Actor_00"):
    x, sr = librosa.load(path)
    plt.figure(figsize=(8, 4))
    # librosa.display.waveplot(x, sr=sr)
    # plt.title('Waveplot - ' + name)
    # plt.show()
    Path('plots' + "/" + folder ).mkdir(parents=True, exist_ok=True)

    # plt.savefig('plots' + "/" + folder + "/WP_" + name.split('.')[0])

    spectrogram = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=128, fmax=8000)
    spectrogram = librosa.power_to_db(spectrogram)

    librosa.display.specshow(spectrogram, y_axis='mel', fmax=8000, x_axis='time');
    # plt.title('Mel Spectrogram - ' + name)
    plt.savefig('plots/' + "/" + folder + "/" + name.split('.')[0] + '.png')
    plt.close()
    # plt.colorbar(format='%+2.0f dB')
    # plt.show()


def plots():
    generate_plots('assets/MaleNeutral.wav', "Male_Neutral")
    generate_plots('assets/FemaleCalm.wav', 'Female_Calm')
    generate_plots('assets/MaleAngry.wav', 'Male_Angry')
    generate_plots('assets/FemaleFearful.wav', 'Female_Fearful')


def preprocessing():
    path = 'assets/archive/'
    actor_folders = os.listdir(path)
    actor_folders.sort()
    print(actor_folders[0:5])

    emotion = []
    gender = []
    actor = []
    file_path = []
    spec_path = []
    for i in actor_folders:
        if i == ".DS_Store":
            continue
        filename = os.listdir(path + i)  # iterate over Actor folders
        for f in filename:  # go through files in Actor folder
            part = f.split('.')[0].split('-')
            print(len(part))
            if len(part) == 7:
                emotion.append(int(part[2]))
                actor.append(int(part[6]))
                bg = int(part[6])
                if bg % 2 == 0:
                    bg = "female"
                else:
                    bg = "male"
                gender.append(bg)
                file_path.append(path + i + '/' + f)
                generate_plots(path+i+'/'+f, f, i)
                spec_path.append('plots' + "/" + i + "/" + f.split('.')[0] + '.png')

    audio_df = pd.DataFrame(emotion)
    audio_df = audio_df.replace(
        {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'})
    audio_df = pd.concat([pd.DataFrame(gender), audio_df, pd.DataFrame(actor)], axis=1)
    audio_df.columns = ['gender', 'emotion', 'actor']
    audio_df = pd.concat([audio_df, pd.DataFrame(file_path, columns=['path'])], axis=1)
    audio_df = pd.concat([audio_df, pd.DataFrame(spec_path, columns=['img_path'])], axis=1)

    pd.set_option('display.max_colwidth', -1)
    print(audio_df.sample(10))

    with open('plots/audio_df.pkl', 'wb') as f:
        pickle.dump(audio_df, f)
    return audio_df


def extract_featrues(audio_df):
    df = pd.DataFrame(columns=['mel_spectrogram'])

    counter = 0

    for index, path in enumerate(audio_df.path):
        X, sample_rate = librosa.load(path, res_type='kaiser_fast', duration=3, sr=44100, offset=0.5)

        # get the mel-scaled spectrogram (ransform both the y-axis (frequency) to log scale, and the “color” axis (amplitude) to Decibels, which is kinda the log scale of amplitudes.)
        spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128, fmax=8000)
        db_spec = librosa.power_to_db(spectrogram)
        # temporally average spectrogram
        log_spectrogram = np.mean(db_spec, axis=0)

        # Mel-frequency cepstral coefficients (MFCCs)
        #     mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
        #     mfcc=np.mean(mfcc,axis=0)

        # compute chroma energy (pertains to 12 different pitch classes)
        #     chroma = librosa.feature.chroma_stft(y=X, sr=sample_rate)
        #     chroma = np.mean(chroma, axis = 0)

        # compute spectral contrast
        #     contrast = librosa.feature.spectral_contrast(y=X, sr=sample_rate)
        #     contrast = np.mean(contrast, axis= 0)

        # compute zero-crossing-rate (zcr:the zcr is the rate of sign changes along a signal i.e.m the rate at
        #     which the signal changes from positive to negative or back - separation of voiced andunvoiced speech.)
        #     zcr = librosa.feature.zero_crossing_rate(y=X)
        #     zcr = np.mean(zcr, axis= 0)

        df.loc[counter] = [log_spectrogram]
        counter = counter + 1

    print(len(df))
    print(df.head())

    df_combined = pd.concat([audio_df, pd.DataFrame(df['mel_spectrogram'].values.tolist())], axis=1)
    df_combined = df_combined.fillna(0)
    df_combined.drop(columns='path', inplace=True)

    print(df_combined.head())

    with open('plots/labeled_data.pkl', 'wb') as f:
        pickle.dump(df_combined, f)
    return df_combined


def model(df_combined):
    train, test = train_test_split(df_combined, test_size=0.2, random_state=0,
                                   stratify=df_combined[['emotion', 'gender', 'actor']])
    X_train = train.iloc[:, 3:]
    y_train = train.iloc[:, :2].drop(columns=['gender'])
    print(X_train.shape)

    X_test = test.iloc[:, 3:]
    y_test = test.iloc[:, :2].drop(columns=['gender'])
    print(X_test.shape)

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    lb = LabelEncoder()
    y_train = to_categorical(lb.fit_transform(y_train))
    y_test = to_categorical(lb.fit_transform(y_test))

    print(y_test[0:10])
    print(lb.classes_)

    X_train2 = X_train
    y_train2 = y_train
    X_test2 = X_test
    y_test2 = y_test

    X_train = X_train[:, :, np.newaxis]
    X_test = X_test[:, :, np.newaxis]

    print(X_train.shape)
    print(X_test.shape)

    dummy_clf = DummyClassifier(strategy="stratified")
    dummy_clf.fit(X_train2, y_train2)
    DummyClassifier(strategy='stratified')
    dummy_clf.predict(X_test2)
    print(dummy_clf.score(X_test2, y_test2))
    # print(X_train2[:9])

    # clf = tree.DecisionTreeClassifier()
    # clf = clf.fit(X_train2, y_train2)
    # clf.predict(X_test2)
    # print(clf.score(X_test2, y_test2))

    return X_train, X_test, y_train, y_test, X_train2, X_test2, y_train2, y_test2, lb


def tf_model(X_train, X_test, y_train, y_test, lb):
    print(X_train.shape[1])
    # BUILD 1D CNN LAYERS
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(64, kernel_size=(10), activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(
        layers.Conv1D(128, kernel_size=10, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(layers.MaxPooling1D(pool_size=8))
    model.add(layers.Dropout(0.4))
    model.add(layers.Conv1D(128, kernel_size=10, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=8))
    model.add(layers.Dropout(0.4))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(8, activation='sigmoid'))
    opt = keras.optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()

    checkpoint = ModelCheckpoint("best_initial_model.hdf5", monitor='val_accuracy', verbose=1,
                                 save_best_only=True, mode='max', period=1, save_weights_only=True)

    model_history = model.fit(X_train, y_train, batch_size=32, epochs=40, validation_data=(X_test, y_test),
                              callbacks=[checkpoint])

    plt.plot(model_history.history['accuracy'])
    plt.plot(model_history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('Initial_Model_Accuracy.png')
    plt.show()
    # summarize history for loss
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('Initial_Model_loss.png')
    plt.show()

    print("Loss of the model is - ", model.evaluate(X_test, y_test)[0])
    print("Accuracy of the model is - ", model.evaluate(X_test, y_test)[1] * 100, "%")

    # PREDICTIONS
    predictions = model.predict(X_test)
    predictions = predictions.argmax(axis=1)
    predictions = predictions.astype(int).flatten()
    predictions = (lb.inverse_transform((predictions)))
    predictions = pd.DataFrame({'Predicted Values': predictions})

    # ACTUAL LABELS
    actual = y_test.argmax(axis=1)
    actual = actual.astype(int).flatten()
    actual = (lb.inverse_transform(actual))
    actual = pd.DataFrame({'Actual Values': actual})

    # COMBINE BOTH
    finaldf = actual.join(predictions)
    print(finaldf[140:150])

    # CREATE CONFUSION MATRIX OF ACTUAL VS. PREDICTION
    cm = confusion_matrix(actual, predictions)
    plt.figure(figsize=(12, 10))
    cm = pd.DataFrame(cm, index=[i for i in lb.classes_], columns=[i for i in lb.classes_])
    ax = sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title('Confusion Matrix', size=20)
    plt.xlabel('Predicted Labels', size=14)
    plt.ylabel('Actual Labels', size=14)
    plt.savefig('Initial_Model_Confusion_Matrix.png')
    plt.show()

    print(classification_report(actual, predictions,
                                target_names=['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad',
                                              'surprise']))


def hyperparameter_training(df_combined):
    train, test = train_test_split(df_combined, test_size=0.2, random_state=0,
                                   stratify=df_combined[['gender', 'actor']])

    X_train = train.iloc[:, 3:]
    y_train = train.iloc[:, :2].drop(columns=['gender'])
    print(X_train.shape)

    X_test = test.iloc[:, 3:]
    y_test = test.iloc[:, :2].drop(columns=['gender'])
    print(X_test.shape)

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    X_train = X_train[:, :, np.newaxis]
    X_test = X_test[:, :, np.newaxis]

    lb = LabelEncoder()

    y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))

    y_trainHot = np.argmax(y_train, axis=1)
    print(X_train.shape)

    def create_classifier(optimizer=keras.optimizers.Adam(lr=0.0001)):
        model = tf.keras.Sequential()
        model.add(layers.Conv1D(64, kernel_size=(10), activation='relu', input_shape=(X_train.shape[1], 1)))
        model.add(
            layers.Conv1D(128, kernel_size=(10), activation='relu', kernel_regularizer=l2(0.01),
                          bias_regularizer=l2(0.01)))
        model.add(layers.MaxPooling1D(pool_size=(8)))
        model.add(layers.Dropout(0.4))
        model.add(layers.Conv1D(128, kernel_size=(10), activation='relu'))
        model.add(layers.MaxPooling1D(pool_size=(8)))
        model.add(layers.Dropout(0.4))
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.4))
        model.add(layers.Dense(8, activation='sigmoid'))
        opt = keras.optimizers.Adam(lr=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    classifier = KerasClassifier(build_fn=create_classifier)
    params = {
        'batch_size': [30, 32, 34],
        'nb_epoch': [25, 50, 75],
        'optimizer': ['adam', 'SGD']}

    grid_search = GridSearchCV(estimator=classifier,
                               param_grid=params,
                               scoring='accuracy',
                               cv=5)

    grid_search = grid_search.fit(X_train, y_trainHot)

    print(grid_search.best_params_)

    print(grid_search.best_score_)


def transfer_vgg16(x_train, x_test, y_train, y_test):
    """
    Intent: Turn the voice snippets into spectrogram - images and train on those
    :param X_train: train data
    :param X_test: test data
    :param y_train: train labels
    :param y_test: test labels
    :return:
    """
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(244, 244, 3))
    for layer in vgg.layers:
        layer.trainable = False

    x = vgg.output
    x = Flatten()(x)  # Flatten dimensions to for use in FC layers
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)  # Dropout layer to reduce overfitting
    x = Dense(256, activation='relu')(x)
    x = Dense(8, activation='softmax')(x)  # Softmax for multiclass
    transfer_model = Model(inputs=vgg.input, outputs=x)

    for i, layer in enumerate(vgg.layers):
        print(i, layer.name, layer.trainable)

    keras.utils.plot_model(transfer_model, show_shapes=True)
    plt.show()

    learning_rate = 5e-5
    transfer_model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(lr=learning_rate),
                           metrics=["accuracy"])
    history = transfer_model.fit(x_train, y_train, batch_size=1, epochs=50, validation_data=(x_test, y_test))
    # PRINT MODEL HISTORY OF ACCURACY AND LOSS OVER EPOCHS
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('Transfer_Model_Accuracy.png')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('Transfer_Model_Loss.png')
    plt.show()


def transform_image_data(df_combined):
    # print(df_combined.columns)
    img_df = df_combined[['gender', 'emotion', 'actor', 'img_path']].copy()
    print(img_df.head)
    x_data = []
    y_data = []
    for index, data in img_df.iterrows():
        image = tf.keras.preprocessing.image.load_img(data['img_path'], color_mode='rgb', target_size=(224,224))
        image = np.array(image)
        # print(data['emotion'])
        x_data.append(image)
        y_data.append(data['emotion'])

    x_train, x_test = train_test_split(x_data, test_size=0.2, random_state=0)
    y_train, y_test = train_test_split(y_data, test_size=0.2, random_state=0)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = np.array(x_train) / 255
    x_test = np.array(x_test) / 255
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    lb = LabelEncoder()
    y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    plt.style.use('fivethirtyeight')
    # audio_df = preprocessing()
    # audio_df.emotion.value_counts().plot(kind='bar')
    # plt.show()
    # data = extract_featrues(audio_df)
    # preprocessing()
    data = pickle.load(open("plots/labeled_data.pkl", 'rb'))
    # x_train, x_test, y_train, y_test = transform_image_data(data)
    # transfer_vgg16(x_train, x_test, y_train, y_test)
    X_train, X_test, y_train, y_test, X_train2, X_test2, y_train2, y_test2, lb = model(data)
    tf_model(X_train, X_test, y_train, y_test, lb)
    # hyperparameter_training(data)
