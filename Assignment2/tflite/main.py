import tensorflow as tf

if __name__ == '__main__':
    # convert keras model -> tflite model
    model = tf.keras.models.load_model('../tensorflow/tensorflow_LeNet5.hdf5')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.inference_type = tf.uint8
    converter.inference_type = tf.float16
    tflite_model = converter.convert()
    # open('tflite_uint8_LeNet5.tflite', 'wb').write(tflite_model)
    open('tflite_float16_LeNet5.tflite', 'wb').write(tflite_model)
    print('Convert Tensorflow model to TFLite model successfully.')

    # interpreter tflite model
    # interpreter = tf.lite.Interpreter(model_path='tflite_LeNet5.tflite')
    # interpreter.allocate_tensors()



