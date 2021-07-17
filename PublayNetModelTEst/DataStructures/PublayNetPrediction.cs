using Microsoft.ML.Data;
using System;
using System.Collections.Generic;

namespace PublayNetModelTEst.DataStructures
{
    public class PublayNetPrediction
    {
        //public readonly string[] Categories2Labels = new string[] { "bg", "text", "title", "list", "table", "figure" };

        [ColumnName("width")]
        public float ImageWidth { get; set; }

        [ColumnName("height")]
        public float ImageHeight { get; set; }

        [ColumnName("boxes")]
        public float[] Boxes { get; set; }

        [ColumnName("labels")]
        public long[] Labels { get; set; }

        [ColumnName("scores")]
        public float[] Scores { get; set; }

        [ColumnName("masks")]
        public float[] Masks { get; set; }

        public IEnumerable<PublayNetResult> Process(float scoreThreshold = 0.7f)
        {
            // check outputs sizes
            if (Labels.Length != Scores.Length)
            {
                throw new ArgumentOutOfRangeException();
            }

            if (Boxes.Length != Labels.Length * 4)
            {
                throw new ArgumentOutOfRangeException();
            }

            if (Masks.Length != Labels.Length * PublayNetBitmapData.NewSize * PublayNetBitmapData.NewSize)
            {
                throw new ArgumentOutOfRangeException();
            }

            // compute scaling and pading
            float padx = 0;
            float padx2 = PublayNetBitmapData.NewSize;
            float pady = 0;
            float pady2 = PublayNetBitmapData.NewSize;

            if (ImageHeight > ImageWidth)
            {
                padx = PublayNetBitmapData.NewSize / 2f - (ImageWidth * PublayNetBitmapData.NewSize / ImageHeight) / 2f;
                padx2 = PublayNetBitmapData.NewSize / 2f + (ImageWidth * PublayNetBitmapData.NewSize / ImageHeight) / 2f;
            }
            else
            {
                pady = PublayNetBitmapData.NewSize / 2f - (ImageHeight * PublayNetBitmapData.NewSize / ImageWidth) / 2f;
                pady2 = PublayNetBitmapData.NewSize / 2f + (ImageHeight * PublayNetBitmapData.NewSize / ImageWidth) / 2f;
            }

            float scaleX = ImageWidth / (padx2 - padx);
            float scaleY = ImageHeight / (pady2 - pady);

            for (int r = 0; r < Labels.Length; r++)
            {
                var score = Scores[r];
                if (score < scoreThreshold) continue;

                // adjust bbox to page size
                var x1 = (Boxes[r * 4] - padx) * scaleX;
                var y1 = (Boxes[r * 4 + 1] - pady) * scaleY;
                var x2 = (Boxes[r * 4 + 2] - padx) * scaleX;
                var y2 = (Boxes[r * 4 + 3] - pady) * scaleY;

                // clip bbox to page bounds
                x1 = Math.Max(x1, 0);
                y1 = Math.Max(y1, 0);
                x2 = Math.Min(x2, ImageWidth - 1);
                y2 = Math.Min(y2, ImageHeight - 1);

                if (x1 > x2 || y1 > y2) continue; // remove invalid bbox

                var label = Labels[r];

                // get mask
                // TODO

                yield return new PublayNetResult(new float[] { x1, y1, x2, y2 }, score, (PublayNetCategories)label); // Categories2Labels[label]
            }
        }
    }
}
