# common packages
from .config import config
from keras.engine import Layer
from keras.layers import SpatialDropout1D, Bidirectional, Dense, LSTM
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, Conv1D
from keras.layers import concatenate
from keras.layers import Input, Embedding, Concatenate
from keras.models import Model
from keras import backend as K
from keras import initializers, optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.wrappers.scikit_learn import KerasClassifier
from Medical_Sieve_Model_Pipeline.dataprocessing import processor as proc
import tensorflow as tf
tf.compat.v1.set_random_seed(config.SEED_VALUE)
# model 1 only
from keras.engine import InputSpec
from keras.layers import Dropout, Lambda
# model 2 only
from keras import regularizers, constraints


# attention for model 1
class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(** kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None

# model1 definition
def pooled_rnn_aspect_model(max_sequence_len=config.MAX_SEQUENCE_LEN,
                            embedding_dim=config.EMBEDDING_DIM, 
                            target_dim=len(config.ASPECT_TARGET), 
                            embedding_matrix=[], 
                            verbose=True, 
                            compile=True):
    recurrent_units = 64
    input_layer = Input(shape=(max_sequence_len,))
    embedding_layer = Embedding(len(embedding_matrix),
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_sequence_len,
                                trainable=False)(input_layer)
    embedding_layer = SpatialDropout1D(0.25)(embedding_layer)

    rnn_1 = Bidirectional(LSTM(recurrent_units, return_sequences=True))(embedding_layer)
    rnn_2 = Bidirectional(LSTM(recurrent_units, return_sequences=True))(rnn_1)
    x = concatenate([rnn_1, rnn_2], axis=2)
    last = Lambda(lambda t: t[:, -1], name='last')(x)
    maxpool = GlobalMaxPooling1D()(x)
    attn = AttentionWeightedAverage()(x)
    average = GlobalAveragePooling1D()(x)

    all_views = concatenate([last, maxpool, average, attn], axis=1)

    x = Dropout(0.5)(all_views)
    x = Dense(144, activation="relu")(x)
    output_layer = Dense(target_dim, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    adam_optimizer = optimizers.Adam(lr=1e-3, decay=1e-6, clipvalue=5)
    model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
    return model

# attention for model 2
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        assert len(input_shape) == 3
        
        self.W = self.add_weight((input_shape[-1],),
                                 initializer = self.init,
                                 name = '{}_W'.format(self.name),
                                 regularizer = self.W_regularizer,
                                 constraint = self.W_constraint)
        self.features_dim = input_shape[-1]
        
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer = 'zero',
                                     name = '{}_b'.format(self.name),
                                     regularizer = self.b_regularizer,
                                     constraint = self.b_constraint)
        else:
            self.b = None
        
        self.built = True
    
    def compute_mask(self, input, input_mask=None):
        return None
    
    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        
        e_ij = K.reshape(K.dot(
                            K.reshape(x, (-1, features_dim)),
                            K.reshape(self.W, (features_dim, 1))
                    ), (-1, step_dim))
        if self.bias:
            e_ij += self.b
        
        e_ij = K.tanh(e_ij)
        
        a = K.exp(e_ij)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(),
                    K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim

# model2 definition
def pooled_rnn_text_cnn_aspect_model(max_sequence_len=config.MAX_SEQUENCE_LEN,
                                    embedding_dim=config.EMBEDDING_DIM, 
                                    target_dim=len(config.ASPECT_TARGET), 
                                    embedding_matrix=[], 
                                    verbose=True, 
                                    compile=True):
    sequence_input = Input(shape=(max_sequence_len,), dtype='int32')
    embedding_layer = Embedding(len(embedding_matrix),
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_sequence_len,
                                trainable=False)
    
    x = embedding_layer(sequence_input)
    x = SpatialDropout1D(0.25)(x)        
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    
    attention = Attention(max_sequence_len)(x)
    conv = Conv1D(64, kernel_size = 3, 
                      padding = "valid", 
                      kernel_initializer = "he_uniform")(x)
    avg_pool1 = GlobalAveragePooling1D()(conv)
    max_pool1 = GlobalMaxPooling1D()(conv)
    avg_pool2 = GlobalAveragePooling1D()(x)
    max_pool2 = GlobalMaxPooling1D()(x)

    x = concatenate([attention, 
                     avg_pool1, 
                     max_pool1, 
                     avg_pool2, 
                     max_pool2])
    
    preds = Dense(target_dim, activation='sigmoid')(x)
    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='Adam',
                  metrics=['acc'])
    return model


def ensemble_aspect_model(input_dim=config.ENSEMBLE_INPUT_DIM, 
                   output_dim=len(config.ASPECT_TARGET), 
                   verbose = True, 
                   compile = True):
    input_layer = Input(shape=(input_dim,))
    x = Dense(input_dim)(input_layer)
    output_layer = Dense(output_dim, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    adam_optimizer = optimizers.Adam(lr=1e-3, decay=1e-6, clipvalue=5)
    model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
    return model

es = EarlyStopping(monitor='val_loss', 
                   mode='min', 
                   verbose=1, 
                   patience=5)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.3, 
                              patience=3, 
                              mode='min', 
                              verbose=1)

checkpointer1 = ModelCheckpoint(filepath=config.MODEL1_PATH, 
                               verbose=1, 
                               save_best_only=True)

checkpointer2 = ModelCheckpoint(filepath=config.MODEL2_PATH, 
                               verbose=1, 
                               save_best_only=True)

checkpointer4 = ModelCheckpoint(filepath=config.MODEL4_PATH, 
                               verbose=1, 
                               save_best_only=True)

pooled_rnn_aspect_clf_for_fold = KerasClassifier(build_fn=pooled_rnn_aspect_model,
                                                 batch_size=config.MODEL1_BATCH_SIZE,
                                                 epochs=config.MODEL1_EPOCHS,
                                                 verbose=1,  # progress bar - required for CI job
                                                 callbacks=[es, reduce_lr]
                                                 )

pooled_rnn_aspect_clf = KerasClassifier(build_fn=pooled_rnn_aspect_model,
                                        batch_size=config.MODEL1_BATCH_SIZE,
                                        epochs=config.MODEL1_EPOCHS,
                                        verbose=1,  # progress bar - required for CI job
                                        callbacks=[es, reduce_lr, checkpointer1]
                                        )

pooled_rnn_text_cnn_aspect_clf_for_fold = KerasClassifier(build_fn=pooled_rnn_text_cnn_aspect_model,
                                                                batch_size=config.MODEL1_BATCH_SIZE,
                                                                epochs=config.MODEL1_EPOCHS,
                                                                verbose=1,  # progress bar - required for CI job
                                                                callbacks=[es, reduce_lr]
                                                                )

pooled_rnn_text_cnn_aspect_clf = KerasClassifier(build_fn=pooled_rnn_text_cnn_aspect_model,
                                                       batch_size=config.MODEL2_BATCH_SIZE,
                                                       epochs=config.MODEL2_EPOCHS,
                                                       verbose=1,  # progress bar - required for CI job
                                                       callbacks=[es, reduce_lr, checkpointer2]
                                                       )


stacking_model_aspect_clf = KerasClassifier(build_fn=ensemble_aspect_model,
                                    batch_size=config.MODEL4_BATCH_SIZE,
                                    epochs=config.MODEL4_EPOCHS,
                                    verbose=1,  # progress bar - required for CI job
                                    callbacks=[es, reduce_lr, checkpointer4]
                                    )


if __name__ == '__main__':
    model1 = pooled_rnn_aspect_model()
    model1.summary()

    model2 = pooled_rnn_text_cnn_aspect_model()
    model2.summary()

    model4 = ensemble_model()
    model4.summary()
