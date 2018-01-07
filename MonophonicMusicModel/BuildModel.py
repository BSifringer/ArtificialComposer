from keras.models import Model
from keras.layers import Input, Dropout, TimeDistributed, Dense, GRU, Masking
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from MonophonicMusicModel.Specifications import *

duration_layers = dict()
pitch_layers = dict()
end_layers = dict()

duration_layers['input'] = Input(shape=(step_size, n_durations), batch_shape=(batch_size, step_size, n_durations))
duration_layers['mask'] = Masking(mask_value=0)(duration_layers['input'])
duration_layers['GRU1'] = GRU(32, return_sequences=True, stateful=True)(duration_layers['mask'])
duration_layers['dropout1'] = Dropout(dropout_rate)(duration_layers['GRU1'])
duration_layers['GRU2'] = GRU(16, return_sequences=True, stateful=True)(duration_layers['dropout1'])
duration_layers['dropout2'] = Dropout(dropout_rate)(duration_layers['GRU2'])
duration_layers['cross'] = TimeDistributed(Dense(16))(duration_layers['dropout2'])

pitch_layers['input'] = Input(shape=(step_size, n_pitches), batch_shape=(batch_size, step_size, n_pitches))
pitch_layers['mask'] = Masking(mask_value=0)(pitch_layers['input'])
pitch_layers['GRU1'] = GRU(64, return_sequences=True, stateful=True)(pitch_layers['mask'])
pitch_layers['dropout1'] = Dropout(dropout_rate)(pitch_layers['GRU1'])
pitch_layers['GRU2'] = GRU(128, return_sequences=True, stateful=True)(pitch_layers['dropout1'])
pitch_layers['dropout2'] = Dropout(dropout_rate)(pitch_layers['GRU2'])
pitch_layers['GRU3'] = GRU(128, return_sequences=True, stateful=True)(pitch_layers['dropout2'])
pitch_layers['dropout3'] = Dropout(dropout_rate)(pitch_layers['GRU3'])
pitch_layers['cross'] = TimeDistributed(Dense(8))(pitch_layers['dropout3'])

duration_layers['merge'] = concatenate([duration_layers['dropout2'], pitch_layers['cross']])
duration_layers['GRU3'] = GRU(32, return_sequences=True, stateful=True)(duration_layers['merge'])
duration_layers['dropout3'] = Dropout(dropout_rate)(duration_layers['GRU3'])
duration_layers['output'] = TimeDistributed(Dense(n_durations, activation='softmax'))(duration_layers['dropout3'])

pitch_layers['merge'] = concatenate([pitch_layers['dropout3'], duration_layers['cross']])
pitch_layers['GRU4'] = GRU(128, return_sequences=True, stateful=True)(pitch_layers['merge'])
pitch_layers['dropout4'] = Dropout(dropout_rate)(pitch_layers['GRU4'])
pitch_layers['GRU5'] = GRU(64, return_sequences=True, stateful=True)(pitch_layers['dropout4'])
pitch_layers['dropout5'] = Dropout(dropout_rate)(pitch_layers['GRU5'])
pitch_layers['shortcut'] = concatenate([pitch_layers['dropout5'], pitch_layers['dropout3'], duration_layers['output']])
pitch_layers['output'] = TimeDistributed(Dense(n_pitches+1, activation='softmax'))(pitch_layers['shortcut'])

model = Model(inputs=[duration_layers['input'], pitch_layers['input']],
              outputs=[duration_layers['output'], pitch_layers['output']])

# end_layers['merge'] = concatenate([pitch_layers['dropout5'], duration_layers['output']])
# end_layers['reduce'] = TimeDistributed(Dense(16))(end_layers['merge'])
# end_layers['GRU1'] = GRU(8, return_sequences=True, stateful=True)(end_layers['reduce'])
# end_layers['GRU2'] = GRU(2, return_sequences=True, stateful=True)(end_layers['GRU1'])
# end_layers['output'] = TimeDistributed(Dense(2, activation='softmax'))(end_layers['GRU2'])

# model = Model(inputs=[duration_layers['input'], pitch_layers['input']],
#               outputs=[duration_layers['output'], pitch_layers['output'], end_layers['output']])

optimizer = Adam(lr=1e-3, clipnorm=1)

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

model.save('models\\'+model_name+'.h5')
