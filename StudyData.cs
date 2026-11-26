
using Microsoft.ML.Data;

public class StudyData
{
    [LoadColumn(0)]
    public float StudyHours { get; set; }

    [LoadColumn(1)]
    public float Score { get; set; }
}
