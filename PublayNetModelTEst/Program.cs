using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using PublayNetModelTEst.DataStructures;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using static Microsoft.ML.Transforms.Image.ImageResizingEstimator;

namespace PublayNetModelTEst
{
    // https://github.com/PaddlePaddle/PaddleOCR/blob/0850586667308d38e113447e8d095e955092fe53/ppstructure/layout/predict_layout.py#L66
    // https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/postprocess/picodet_postprocess.py#L98

    class Program
    {
        float[] strides = new float[] { 8, 16, 32, 64 };
        static readonly Dictionary<PublayNetCategories, Color> Categories2Colors = new Dictionary<PublayNetCategories, Color>()
        {
            { PublayNetCategories.Background,   Color.AliceBlue },
            { PublayNetCategories.Text,         Color.Red },
            { PublayNetCategories.Title,        Color.Green },
            { PublayNetCategories.List,         Color.Yellow },
            { PublayNetCategories.Table,        Color.Brown },
            { PublayNetCategories.Figure,       Color.Orange }
        };

        // Unzip the model first!
        const string modelLocation = @"C:\Users\Bob\Document Layout Analysis\PaddlePaddle\picodet_lcnet_x1_0_fgd_layout_infer\onnx\picodet_lcnet_x1_0_fgd_layout_infer.onnx";

        const string imageFolder = @"Assets\Images";

        const string imageOutputFolder = @"Assets\Output";

        static void Main()
        {
            Directory.CreateDirectory(imageOutputFolder);

            MLContext mlContext = new MLContext();
            mlContext.Log += MlContext_Log;

            // Define scoring pipeline
            var pipeline = mlContext.Transforms.ResizeImages(inputColumnName: "bitmap",
                                                             outputColumnName: "image",
                                                             imageWidth: PublayNetBitmapData.NewWidth,
                                                             imageHeight: PublayNetBitmapData.NewHeight,
                                                             resizing: ResizingKind.IsoPad)
                            .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "image",
                                                    scaleImage: 1f / 255f,
                                                    orderOfExtraction: ImagePixelExtractingEstimator.ColorsOrder.ARGB))
                            .Append(mlContext.Transforms.ApplyOnnxModel(inputColumnNames: new[] { "image" },
                                                                        outputColumnNames: new[]
                                                                        {
                                                                            "transpose_0.tmp_0",
                                                                            "transpose_2.tmp_0",
                                                                            "transpose_4.tmp_0",
                                                                            "transpose_6.tmp_0",

                                                                            "transpose_1.tmp_0",
                                                                            "transpose_3.tmp_0",
                                                                            "transpose_5.tmp_0",
                                                                            "transpose_7.tmp_0"
                                                                        },
                                                                        modelFile: modelLocation)); //gpuDeviceId: 0, fallbackToCpu: true,          

            // Fit on empty list to obtain input data schema
            var model = pipeline.Fit(mlContext.Data.LoadFromEnumerable(new List<PublayNetBitmapData>()));

            // Create prediction engine
            var predictionEngine = mlContext.Model.CreatePredictionEngine<PublayNetBitmapData, PublayNetPrediction>(model);

            // save model
            mlContext.Model.Save(model, predictionEngine.OutputSchema, Path.ChangeExtension(modelLocation, "mlnet.zip"));
            var modelLoaded = mlContext.Model.Load(Path.ChangeExtension(modelLocation, "mlnet.zip"), out var schema);
            Stopwatch stopWatch = new Stopwatch();
            foreach (var imagePath in Directory.GetFiles(imageFolder))
            {
                Console.WriteLine($"Processing {imagePath}...");

                using (var bitmap = MLImage.CreateFromFile(imagePath)) //new Bitmap(Image.FromFile(imagePath)))
                {
                    // predict
                    stopWatch.Start();
                    var prediction = predictionEngine.Predict(new PublayNetBitmapData() { Image = bitmap });

                    const int num_outs = 4;
                    var np_score_list = new float[num_outs][] // scores
                    {
                        prediction.t0,
                        prediction.t2,
                        prediction.t4,
                        prediction.t6,
                    };

                    var np_boxes_list = new float[num_outs][] // raw_boxes
                    {
                        prediction.t1,
                        prediction.t3,
                        prediction.t5,
                        prediction.t7,
                    };

                    //prediction.t2.GetValues().Slice()



                    var results = prediction.Process(0.5f);
                    stopWatch.Stop();
                    Console.WriteLine($"Done in {stopWatch.Elapsed.TotalSeconds:0.00}s.");
                    stopWatch.Reset();

                    prediction = null;
                    GC.Collect();

                    // draw predictions
                    /*
                    using (var g = Graphics.FromImage(bitmap))
                    {
                        foreach (var result in results)
                        {
                            var x1 = result.Bbox[0];
                            var y1 = result.Bbox[1];
                            var w = result.Bbox[2] - x1;
                            var h = result.Bbox[3] - y1;

                            using (var pen = new Pen(Categories2Colors[result.Category]))
                            {
                                g.DrawRectangle(pen, x1, y1, w, h);
                            }

                            using (var brushes = new SolidBrush(Color.FromArgb(50, Categories2Colors[result.Category])))
                            {
                                g.FillRectangle(brushes, x1, y1, w, h);
                            }

                            g.DrawString(result.Category + " " + result.Score.ToString("0.00"),
                                         new Font("Arial", 12), Brushes.Blue, new PointF(x1, y1));
                        }

                        bitmap.Save(Path.Combine(imageOutputFolder, Path.GetFileName(imagePath)));
                    }
                    */
                }
            }
            

            predictionEngine.Dispose();
            model.Dispose();

            Console.WriteLine("Press any key to exit.");
            Console.ReadKey();
        }

        private static void MlContext_Log(object sender, LoggingEventArgs e)
        {
            Console.WriteLine($"{e.Kind}: {e.RawMessage}");
        }
    }
}
