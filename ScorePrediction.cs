
using Microsoft.ML.Data;

public class ScorePrediction
{
    [ColumnName("Score")]
    public float PredictedScore { get; set; }
}
