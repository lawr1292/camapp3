using System.Collections.Generic;
using Android.Graphics;
using Java.Lang;
using Xamarin.TensorFlow.Lite;
using Org.Tensorflow.Lite.Support.Image;

namespace camapp3
{
    //
    // Helper class used to communicate between our app and the TF object detection model
    //
    public class ObjectDetectionHelper
    {
        // Abstraction object that wraps a prediction output in an easy to parse way
        public class ObjectPrediction
        { public RectF Location; public string Label; public float Score; }

        private float[][][] locations = new float[1][][] { new float[ObjectCount][] };
        private float[][] labelIndices = new float[1][] { new float[ObjectCount] };
        private float[][] scores = new float[1][] { new float[ObjectCount] };

        private Object Locations, LabelIndices, Scores;
        private IDictionary<Integer, Object> outputBuffer;

        /*
         * Initializes the ObjectDetectionHelper with a tflite Interpreter and a list of labels
         */
        public ObjectDetectionHelper(Interpreter tflite, IList<string> labels)
        {
             
            this.tflite = tflite;
            this.labels = labels;

            // set up the two day array that will contain the cooridnates of the face
            for (int i = 0; i < ObjectCount; i++)
            {
                locations[0][i] = new float[4];
            }

            // Convert each supposed out to java objects
            // because the tensorflow library outputs in the form of java objects
            Locations = Object.FromArray(locations);
            LabelIndices = Object.FromArray(labelIndices);
            Scores = Object.FromArray(scores);

            // create a dictionary that creates a Key_Value pair for each java object
            // this dictionary will be what "catches" the tensorflow output object
            outputBuffer = new Dictionary<Integer, Object>()
            {
                [new Integer(0)] = Locations,
                [new Integer(1)] = LabelIndices,
                [new Integer(2)] = Scores,
                [new Integer(3)] = new float[1],
            };
        }

        private Interpreter tflite;
        private IList<string> labels;

        // Parse the output from a series of array to a readable "objectPredictions" object
        private ObjectPrediction[] Predictions()
        {
            var objectPredictions = new ObjectPrediction[ObjectCount];
            for (int i = 0; i < ObjectCount; i++)
            {
                objectPredictions[i] = new ObjectPrediction
                {
                    // The locations are an array of [0, 1] floats for [top, left, bottom, right]
                    Location = new RectF(
                        locations[0][i][1], locations[0][i][0],
                        locations[0][i][3], locations[0][i][2]),

                    Label = labels[(int)labelIndices[0][i]],

                    // Score is a single value of [0, 1]
                    Score = scores[0][i]
                };
            }
            return objectPredictions;
        }

        // Method that is called by the MainActivity
        public ObjectPrediction[] Predict(TensorImage image)
        {
            // Runs the interpreter
            // Multiple outputs are required in this case?
            tflite.RunForMultipleInputsOutputs(new Object[] { image.Buffer }, outputBuffer);

            // Changes the newly created javaobjects that are contained in the outputBuffer back into multidimensional arrays
            locations = Locations.ToArray<float[][]>();
            labelIndices = LabelIndices.ToArray<float[]>();
            scores = Scores.ToArray<float[]>();

            return Predictions();
        }

        private const int ObjectCount = 10;
    }
}