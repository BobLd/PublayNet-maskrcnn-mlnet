using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using System.Drawing;

namespace PublayNetModelTEst.DataStructures
{
    class PublayNetBitmapData
    {
        /// <summary>
        /// Image new size after resizing step.
        /// </summary>
        public const int NewSize = 1300;

        [ColumnName("bitmap")]
        [ImageType(NewSize, NewSize)]
        public Bitmap Image { get; set; }

        [ColumnName("width")]
        public float ImageWidth => Image.Width;

        [ColumnName("height")]
        public float ImageHeight => Image.Height;
    }
}
