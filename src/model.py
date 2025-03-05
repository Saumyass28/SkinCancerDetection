import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class SkinCancerClassifier:
    def __init__(self, input_shape=(224, 224, 3), num_classes=1):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def build_model(self, learning_rate=1e-4):
        """
        Build CNN model using ResNet50V2 with transfer learning
        """
        base_model = ResNet50V2(
            weights='imagenet', 
            include_top=False, 
            input_shape=self.input_shape
        )
        base_model.trainable = False
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        output = Dense(
            self.num_classes, 
            activation='sigmoid', 
            name='classification_layer'
        )(x)
        
        model = Model(inputs=base_model.input, outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=20):
        """
        Train the model with early stopping
        """
        model = self.build_model()
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=[early_stopping]
        )
        
        # Quantize model for faster inference
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quantized_model = converter.convert()
        
        # Save models
        model.save('models/skin_cancer_model.h5')
        with open('models/quantized_model.tflite', 'wb') as f:
            f.write(quantized_model)
        
        return model, history

if __name__ == "__main__":
    classifier = SkinCancerClassifier()