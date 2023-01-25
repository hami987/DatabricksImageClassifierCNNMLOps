import mlflow
from models.cnn import build_model, SklearnModelWrapper
from data.preprocessing import download_preprocess
from data.plot_data import plot_10
import tensorflow as tf

def inference(model, test_images, test_labels):
    # Evaluate the model on the test data using `evaluate`
    cnn=model.compile(optimizer='adam', 
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    print("Evaluate on test data")
    results = model.evaluate(test_images,test_labels, batch_size=128)
    print("test loss, test acc:", results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    print("Generate predictions for 3 samples")
    predictions = model.predict(test_images[:3])
    print("predictions shape:", predictions.shape)

if __name__ == "__main__":
    
    model = build_model()
    train_images, train_labels, test_images, test_labels, class_names = download_preprocess()
    #plot_10(train_images=train_images, class_names=class_names, train_labels=train_labels)
    
    with mlflow.start_run(run_name='cnn_image_classifier_demo'):
        cnn=model.compile(optimizer='adam', 
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        history = model.fit(train_images, train_labels, epochs=10, 
                            validation_data=(test_images, test_labels))
        mlflow.log_metric('accuracy', history.history['accuracy'][-1])
        wrappedModel = SklearnModelWrapper(model)
        mlflow.pyfunc.log_model("cnn_image_classifier_demo", python_model=wrappedModel)
    
    