using System.IO;
using System.Linq;
using Android;
using Android.App;
using Android.Graphics;
using Android.OS;
using Android.Runtime;
using Android.Util;
using Android.Views;
using Android.Widget;
using AndroidX.AppCompat.App;
using AndroidX.Camera.Core;
using AndroidX.Camera.Lifecycle;
using AndroidX.Camera.View;
using AndroidX.ConstraintLayout.Widget;
using AndroidX.Core.App;
using AndroidX.Core.Content;
using Java.Lang;
using Java.Nio;
using Java.Util.Concurrent;
using Xamarin.TensorFlow.Lite;
using Org.Tensorflow.Lite.Support.Common.Ops;
using Org.Tensorflow.Lite.Support.Image;
using Org.Tensorflow.Lite.Support.Image.Ops;
using Org.Tensorflow.Lite.Support.Common;
using Org.Tensorflow.Lite.Support.Metadata;

namespace camapp3
{
    [Activity(Label = "@string/app_name", Theme = "@style/AppTheme", MainLauncher = true)]
    public class MainActivity : AppCompatActivity,
        PixelCopy.IOnPixelCopyFinishedListener,
        ImageAnalysis.IAnalyzer
    {

        private ConstraintLayout container;
        private Bitmap bitmapBuffer;

        private IExecutorService executor = Executors.NewSingleThreadExecutor();

        private int lensFacing = CameraSelector.LensFacingFront;
        private bool isFrontFacing() { return lensFacing == CameraSelector.LensFacingFront; }

        private bool pauseAnalysis = false;
        private bool sendPhoto = false;
        private int imageRotationDegrees = 0;
        private TensorImage tfImageBuffer = new TensorImage(Xamarin.TensorFlow.Lite.DataType.Uint8);

        private ImageProcessor tfImageProcessor;
        private Interpreter tflite;
        private ObjectDetectionHelper detector;
        private Size tfInputSize;
        private Stream labelFile;
        private const string ModelPath = "face_detection_short.tflite";

        private const string TAG = "CameraMediapipe";
        private const int REQUEST_CODE_PERMISSIONS = 10;
        private const string FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS";

        ImageCapture imageCapture;
        Java.IO.File outputDirectory;
        IExecutor cameraExecutor;

        private bool[] correctEyes = new bool[30];
        int eyeIter = 0;
        private int frameCounter;
        private long lastFpsTimestamp;
        private SurfaceView surfaceView;
        private TextureView textureView;


        PreviewView viewFinder;

        protected override void OnCreate(Bundle savedInstanceState)
        {
            base.OnCreate(savedInstanceState);
            Xamarin.Essentials.Platform.Init(this, savedInstanceState);
            // Set our view from the "main" layout resource
            SetContentView(Resource.Layout.activity_main);
            container = FindViewById(Resource.Id.camera_container) as ConstraintLayout;
            this.viewFinder = this.FindViewById<PreviewView>(Resource.Id.viewFinder);
            var camera_capture_button = this.FindViewById<Button>(Resource.Id.camera_capture_button);


            outputDirectory = GetOutputDirectory();
            // ~~~~~~~~~~~~~~~~~~~~~~~~
            // ML model variables
            // ~~~~~~~~~~~~~~~~~~~~~~~~

            ByteBuffer tfliteModel = FileUtil.LoadMappedFile(this, ModelPath);
            //MetadataExtractor metadataExtractor = new MetadataExtractor(tfliteModel);
            //Stream labelFile = metadataExtractor.GetAssociatedFile("labelmap.txt");
            var labelFile = new System.Collections.Generic.List<string>();
            labelFile.Add("Face");
            labelFile.Add("Not Face");

            tflite = new Interpreter(tfliteModel);
            detector = new ObjectDetectionHelper(tflite,
                labelFile);

            var inputIndex = 0;
            var tensor = tflite.GetInputTensor(inputIndex);
            var shape = tensor.Shape();
            var width = shape[1];
            var height = shape[2];
            tfInputSize = new Size(height, width);

            // Request camera permissions
            string[] permissions = new string[] { Manifest.Permission.Camera, Manifest.Permission.WriteExternalStorage };
            if (permissions.FirstOrDefault(x => ContextCompat.CheckSelfPermission(this, x) != Android.Content.PM.Permission.Granted) != null)
            {
                ActivityCompat.RequestPermissions(this, permissions, REQUEST_CODE_PERMISSIONS);
                ActivityCompat.RequestPermissions(this, permissions, REQUEST_CODE_PERMISSIONS);
            }
            else
                StartCamera();

            // Set up the listener for take photo button
            camera_capture_button.SetOnClickListener(new OnClickListener(() => OnClick()));
            viewFinder.Touch += ViewFinderOnTouch;
            cameraExecutor = Executors.NewSingleThreadExecutor();
        }
        protected override void OnDestroy()
        {
            // nnApiDelegate?.Close();
            base.OnDestroy();
            cameraExecutor.Dispose();
        }
        public override void OnRequestPermissionsResult(int requestCode, string[] permissions, [GeneratedEnum] Android.Content.PM.Permission[] grantResults)
        {
            if (requestCode == REQUEST_CODE_PERMISSIONS)
            {
                if (permissions.FirstOrDefault(x => ContextCompat.CheckSelfPermission(this, x) != Android.Content.PM.Permission.Granted) == null)
                {
                    StartCamera();
                }
                else
                {
                    Toast.MakeText(this, "Permissions not Granted by the user.", ToastLength.Short).Show();
                    this.Finish();
                    return;
                }

            }
        }

        public void OnClick()
        {
            // Disable all Camera Controls
            var v = FindViewById<Button>(Resource.Id.camera_capture_button);
            v.Enabled = false;

            ImageView imagePredicted = FindViewById(Resource.Id.image_predicted) as ImageView;
            if (sendPhoto)
            {

                var filename = System.IO.Path.Combine(outputDirectory.Path, "myfile.jpg");
                Android.Graphics.Drawables.BitmapDrawable bd = (Android.Graphics.Drawables.BitmapDrawable)imagePredicted.Drawable;
                Android.Graphics.Bitmap bitmap = bd.Bitmap;
                SaveImage(bitmap, filename);
            }

            if (pauseAnalysis)
            {
                // If image analysis is in paused state, resume it
                pauseAnalysis = false;
                imagePredicted.Visibility = ViewStates.Gone;
            }
            else
            {
                // Otherwise, pause image analysis and freeze image
                pauseAnalysis = true;
                var matrix = new Matrix();
                matrix.PostRotate((float)imageRotationDegrees);
                var uprightImage = Bitmap.CreateBitmap(
                    bitmapBuffer, 0, 0, bitmapBuffer.Width, bitmapBuffer.Height, matrix, false);
                imagePredicted.SetImageBitmap(uprightImage);
                imagePredicted.Visibility = ViewStates.Visible;
            }

            // Re-enable camera controls
            v.Enabled = true;
        }

        private void StartCamera()
        {

            var CameraProviderFuture = ProcessCameraProvider.GetInstance(this);

            viewFinder.Post(() =>
            {
                CameraProviderFuture.AddListener(new Runnable(() =>
                {
                    // Binds the lifecycle of cameras to the lifecycle owner
                    var cameraProvider = (ProcessCameraProvider)CameraProviderFuture.Get();

                    // Preview
                    var preview = new Preview.Builder().Build();
                    preview.SetSurfaceProvider(viewFinder.SurfaceProvider);


                    //Take Photo
                    this.imageCapture = new ImageCapture.Builder().Build();

                    Log.Debug(TAG, this.viewFinder.Display.Rotation.ToString());
                    var imageAnalysis = new ImageAnalysis.Builder()
                        .SetTargetAspectRatio(AspectRatio.Ratio43)
                        .SetTargetRotation((int)this.viewFinder.Display.Rotation)
                        .SetBackpressureStrategy(ImageAnalysis.StrategyKeepOnlyLatest).Build();

                    frameCounter = 0;
                    lastFpsTimestamp = JavaSystem.CurrentTimeMillis();

                    imageAnalysis.SetAnalyzer(cameraExecutor, this);

                    // Select Front Camera as default
                    CameraSelector cameraSelector = new CameraSelector.Builder().RequireLensFacing(lensFacing).Build();

                    try
                    {
                        // Unbind use cases before rebinding
                        cameraProvider.UnbindAll();

                        // Bind use cases to camera
                        cameraProvider.BindToLifecycle(this, cameraSelector, preview, imageAnalysis);
                    }
                    catch (Exception ex)
                    {
                        Log.Debug(TAG, "Use Case Binding Failed", ex);
                        Toast.MakeText(this, $"Use case binding failed: {ex.Message}", ToastLength.Short).Show();

                    }

                    // Use the camera object to link our preview use case with the view
                    preview.SetSurfaceProvider(viewFinder.SurfaceProvider);

                    OnPreviewSizeChosen(preview.AttachedSurfaceResolution);

                    viewFinder.Post(() =>
                    {
                        surfaceView = (SurfaceView)viewFinder.GetChildAt(0);
                        textureView = viewFinder.GetChildAt(0) as TextureView;
                    });

                }), ContextCompat.GetMainExecutor(this)); // returns an executor that runs on the main thread
            });
        }

        private void OnPreviewSizeChosen(Size size)
        {
            imageRotationDegrees = viewFinder.Display.Rotation switch
            {
                SurfaceOrientation.Rotation0 => 0,
                SurfaceOrientation.Rotation90 => 270,
                SurfaceOrientation.Rotation180 => 180,
                SurfaceOrientation.Rotation270 => 90,
                _ => 0
            };


            bitmapBuffer = Bitmap.CreateBitmap(size.Height, size.Width, Bitmap.Config.Argb8888);


            var cropSize = Math.Min(bitmapBuffer.Width, bitmapBuffer.Height);
            tfImageProcessor = new ImageProcessor.Builder()
                //.Add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                .Add(new ResizeOp(tfInputSize.Width, tfInputSize.Height, ResizeOp.ResizeMethod.Bilinear))
                .Add(new Rot90Op(-imageRotationDegrees / 90))
                .Add(new NormalizeOp(114.98212f, 59.765034f))
                .Build();
        }

        public void Analyze(IImageProxy image)
        {
            image.Close();


            // Early exit: image analysis is in paused state
            if (pauseAnalysis)
                return;

            // Copy out RGB bits to our shared buffer
            if (surfaceView != null && surfaceView.Holder.Surface != null && surfaceView.Holder.Surface.IsValid)
                PixelCopy.Request(surfaceView, bitmapBuffer, this, surfaceView.Handler);
            else if (textureView != null && textureView.IsAvailable)
                textureView.GetBitmap(bitmapBuffer);

            // Process Image in Tensorflow
            tfImageBuffer.Load(bitmapBuffer);
            var tfImage = (TensorImage)tfImageProcessor.Process(tfImageBuffer);

            // Perform the object detection for the current frame
            var objectPreds = detector.Predict(tfImage);

            // Report only the top prediction
            ReportPrediction(objectPreds.OrderBy(p => p.Score).LastOrDefault());

            // Count fps of pipeline
            var frameCount = 10;
            if (++frameCounter % frameCount == 0)
            {
                frameCounter = 0;
                var now = JavaSystem.CurrentTimeMillis();
                var delta = now - lastFpsTimestamp;
                var fps = 1000 * (float)frameCount / delta;
                Log.Debug(TAG, "FPS: " + fps.ToString("0.00") + " with tensorSize: " +
                    tfImage.Width + " x " + tfImage.Height);
                lastFpsTimestamp = now;
            }

        }

        private void ReportPrediction(ObjectDetectionHelper.ObjectPrediction prediction)
        {
            viewFinder.Post(() =>
            {
                var qtbutton = (Button)FindViewById(Resource.Id.camera_capture_button);
                var textPrediction = (TextView)FindViewById(Resource.Id.text_prediction);
                // Update the text
                textPrediction.Text = prediction.Score.ToString("0.00") + prediction.Label;

                Log.Debug(TAG, // "Prediction is null: " + (prediction == null).ToString() +
                    "\nPrediction.score = " + prediction.Score +
                    "\nPrediction label = " + prediction.Label);
                if (prediction != null && prediction.Label == "Eye")
                {
                    correctEyes[eyeIter] = true;
                    eyeIter++;
                }
                else if (prediction != null && prediction.Label == "Not Eye")
                {
                    correctEyes = new bool[30];
                    eyeIter = 0;
                }

                if (eyeIter == 5)
                {
                    correctEyes = new bool[30];
                    eyeIter = 0;
                    pauseAnalysis = true;
                    qtbutton.Text = "Save Photo?";
                    sendPhoto = true;
                    ImageView imagePredicted = FindViewById(Resource.Id.image_predicted) as ImageView;

                    var matrix = new Matrix();
                    matrix.PostRotate((float)imageRotationDegrees);
                    var uprightImage = Bitmap.CreateBitmap(
                        bitmapBuffer, 0, 0, bitmapBuffer.Width, bitmapBuffer.Height, matrix, false);
                    imagePredicted.SetImageBitmap(uprightImage);
                    imagePredicted.Visibility = ViewStates.Visible;
                    imagePredicted.Touch += imgOnTouch;
                }
            });
        }

        // Save photos to /Pictures/CameraX/
        private Java.IO.File GetOutputDirectory()
        {
            var mediaDir = Environment.GetExternalStoragePublicDirectory(System.IO.Path.Combine(Environment.DirectoryPictures, Resources.GetString(Resource.String.app_name)));

            if (mediaDir != null && mediaDir.Exists())
                return mediaDir;

            var file = new Java.IO.File(mediaDir, string.Empty);
            file.Mkdirs();
            return file;
        }

        public bool SaveImage(Bitmap bitmap, string filename)
        {
            bool success = false;
            FileStream fs = null;
            try
            {
                using (fs = new FileStream(filename, FileMode.Create))
                {
                    bitmap.Compress(Bitmap.CompressFormat.Jpeg, 8, fs);
                    success = true;
                }
            }
            catch (Exception e)
            {
                System.Console.WriteLine("SaveImage exception: " + e.Message);
            }
            finally
            {
                if (fs != null)
                    fs.Close();
            }
            return success;
        }

        private void imgOnTouch(object sender, View.TouchEventArgs touchEventArgs)
        {
            ((ImageView)sender).Touch -= imgOnTouch;
            ((ImageView)sender).Visibility = ViewStates.Gone;
            pauseAnalysis = false;
        }

        private void ViewFinderOnTouch(object sender, View.TouchEventArgs touchEventArgs)
        {

            bool actTruth = touchEventArgs.Event.Action == MotionEventActions.Down;
            bool dtTruth = touchEventArgs.Event.DownTime > (long)233420000;

            if (actTruth && dtTruth)
            {
                if (lensFacing == CameraSelector.LensFacingFront)
                    lensFacing = CameraSelector.LensFacingBack;
                else
                    lensFacing = CameraSelector.LensFacingFront;

                StartCamera();
            }
        }

        public void OnPixelCopyFinished(int copyResult)
        {
            if (copyResult != (int)PixelCopyResult.Success)
                Log.Error(TAG, "OnPixelCopyFinished() failed");
        }
    }

    public class ObjectPrediction
    {
        public string label { get; set; }
        public float score { get; set; }
    }
}