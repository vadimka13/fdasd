
using Microsoft.ML;
using System;
using System.IO;
using System.Net.Security;

class Program
{
    static void Main(string[] args)
    {
        // 1. Создаем контекст ML.NET
        var mlContext = new MLContext(seed: 0);

        Console.WriteLine("==================================");
        Console.WriteLine("     ML.NET - ЛИНЕЙНАЯ РЕГРЕССИЯ");
        Console.WriteLine("==================================");

        // 2. Загружаем данные
        string dataPath = Path.Combine(
            AppContext.BaseDirectory,
            "data.csv"
        );

        if (!File.Exists(dataPath))
        {
            Console.WriteLine("Ошибка: файл data.csv не найден!");
            return;
        }

        var dataView = mlContext.Data.LoadFromTextFile<StudyData>(
            path: dataPath,
            hasHeader: true,
            separatorChar: ','
        );

        Console.WriteLine("\nДанные успешно загружены!");

        // 3. Разделяем данные на обучающую и тестовую выборки
        var split = mlContext.Data.TrainTestSplit(
            dataView,
            testFraction: 0.2
        );

        var trainData = split.TrainSet;
        var testData = split.TestSet;

        Console.WriteLine("Данные разделены на выборки.");

        // 4. Создаем конвейер обучения
        var pipeline = mlContext.Transforms.Concatenate(
                "Features",
                nameof(StudyData.StudyHours)
            )
            .Append(
                mlContext.Regression.Trainers.Sdca(
                    labelColumnName: nameof(StudyData.Score),
                    maximumNumberOfIterations: 100
                )
            );

        // 5. Обучаем модель
        Console.WriteLine("\nОбучение модели...");

        var model = pipeline.Fit(trainData);

        Console.WriteLine("Модель успешно обучена!");

        // 6. Проверяем качество модели
        var predictions = model.Transform(testData);

        var metrics = mlContext.Regression.Evaluate(
            predictions,
            labelColumnName: nameof(StudyData.Score)
        );

        Console.WriteLine("\n==================================");
        Console.WriteLine("       КАЧЕСТВО МОДЕЛИ");
        Console.WriteLine("==================================");

        Console.WriteLine(
            $"MAE: {metrics.MeanAbsoluteError:F2}"
        );

        Console.WriteLine(
            $"R²: {metrics.RSquared:F2}"
        );

        // 7. Создаем механизм предсказания
        var predictionEngine =
            mlContext.Model.CreatePredictionEngine<
                StudyData,
                ScorePrediction
            >(model);

        // 8. Прогноз для нескольких значений
        Console.WriteLine("\n==================================");
        Console.WriteLine("         ПРИМЕРЫ ПРОГНОЗА");
        Console.WriteLine("==================================");

        float[] hours = { 3, 5, 5.5f, 6, 8, 10 };

        foreach (float hour in hours)
        {
            var input = new StudyData
            {
                StudyHours = hour
            };

            var prediction = predictionEngine.Predict(input);

            Console.WriteLine(
                $"{hour} часов -> прогноз оценки: " +
                $"{prediction.PredictedScore:F1}"
            );
        }

        // 9. Интерактивный ввод пользователя
        Console.WriteLine("\n==================================");
        Console.WriteLine("       СОБСТВЕННЫЙ ПРОГНОЗ");
        Console.WriteLine("==================================");

        while (true)
        {
            Console.Write("\nВведите часы подготовки (или exit): ");

            string? inputText = Console.ReadLine();

            if (string.Equals(
                inputText,
                "exit",
                StringComparison.OrdinalIgnoreCase))
            {
                break;
            }

            if (!float.TryParse(
                inputText,
                System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture,
                out float studyHours))
            {
                Console.WriteLine(
                    "Ошибка! Введите число, например 5.5"
                );

                continue;
            }

            if (studyHours < 0)
            {
                Console.WriteLine(
                    "Количество часов не может быть отрицательным."
                );

                continue;
            }

            var newData = new StudyData
            {
                StudyHours = studyHours
            };

            var result = predictionEngine.Predict(newData);

            Console.WriteLine(
                $"Прогнозируемая оценка: {result.PredictedScore:F2}"
            );
        }

        // 10. Сохраняем обученную модель
        string modelPath = Path.Combine(
            AppContext.BaseDirectory,
            "model.zip"
        );

        mlContext.Model.Save(
            model,
            trainData.Schema,
            modelPath
        );

        Console.WriteLine("\nМодель сохранена: " + modelPath);

        Console.WriteLine("\nРабота программы завершена!");
    }
}
