import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

class SUSAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.sus_score = None
        self.likert_mapping = {
            'Strongly Disagree': 1,
            'Disagree': 2,
            'Neutral': 3,
            'Agree': 4,
            'Strongly Agree': 5
        }

        self.sus_questions = [
            'I think I would like to use this system frequently. ',  # Note the trailing space
            'I found the system unnecessarily complex.',
            'I thought the system was easy to use.',
            'I think that I would need the support of a technical person to be able to use this system.',
            'I found the various functions in this system were well integrated.',
            'I thought there was too much inconsistency in this system.',
            'I would imagine that most people would learn to use this system very quickly.',
            'I found the system very cumbersome to use.',
            'I felt very confident using the system.',
            'I needed to learn a lot of things before I could get going with this system.'
        ]

    def load_data(self):

        try:

            self.df = pd.read_excel(self.file_path)


            print("Available columns in dataset:")
            for col in self.df.columns:
                print(f"- {col}")

            for question in self.sus_questions:
                if question in self.df.columns:
                    self.df[question] = self.df[question].map(self.likert_mapping)
                else:
                    print(f"Warning: Question not found in data: {question}")

            self.calculate_sus_scores()

            return self.df
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def calculate_sus_scores(self):

        try:
            scores = []
            for _, row in self.df.iterrows():
                score = 0
                for i, question in enumerate(self.sus_questions, 1):
                    if question in row.index:
                        if i % 2 == 1:
                            score += (row[question] - 1) * 2.5
                        else:
                            score += (5 - row[question]) * 2.5
                scores.append(score)

            self.df['SUS Score'] = scores
            self.sus_score = np.mean(scores)

        except Exception as e:
            print(f"Error calculating SUS scores: {str(e)}")
            raise

    def generate_visualizations(self, output_dir='.'):
        try:
            os.makedirs(output_dir, exist_ok=True)

            # 1. Overall SUS Score Distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(data=self.df, x='SUS Score', bins=10)
            plt.axvline(x=self.sus_score, color='r', linestyle='--', label=f'Mean: {self.sus_score:.1f}')
            plt.title('Distribution of SUS Scores')
            plt.xlabel('SUS Score')
            plt.ylabel('Count')
            plt.legend()
            plt.savefig(os.path.join(output_dir, 'sus_score_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # 2. Question-wise Response Distribution
            plt.figure(figsize=(15, 8))
            response_means = self.df[self.sus_questions].mean().sort_values()
            colors = ['r' if i % 2 == 1 else 'b' for i in range(len(response_means))]

            # Create shortened labels for better visualization
            short_labels = {q: q[:30] + '...' if len(q) > 30 else q for q in response_means.index}
            response_means.index = [short_labels[q] for q in response_means.index]

            response_means.plot(kind='barh', color=colors)
            plt.title('Mean Responses by Question')
            plt.xlabel('Mean Score')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'question_means.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # 3. Heatmap of Response Distributions
            plt.figure(figsize=(12, 8))
            correlation_matrix = self.df[self.sus_questions].corr()

            # Shorten labels for heatmap
            correlation_matrix.index = [q[:30] + '...' if len(q) > 30 else q for q in correlation_matrix.index]
            correlation_matrix.columns = correlation_matrix.index

            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
            plt.title('Correlation between Questions')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # 4. Box Plot of Responses
            plt.figure(figsize=(15, 6))
            plt.boxplot([self.df[q] for q in self.sus_questions], labels=[q[:30] + '...' if len(q) > 30 else q for q in self.sus_questions])
            plt.xticks(rotation=45, ha='right')
            plt.title('Response Distribution by Question')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'response_boxplot.png'), dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Error generating visualizations: {str(e)}")
            raise

    def generate_report(self):
        """Generate statistical report for SUS analysis."""
        report = {
            'overall_metrics': {
                'mean_sus_score': self.sus_score,
                'median_sus_score': self.df['SUS Score'].median(),
                'std_sus_score': self.df['SUS Score'].std(),
                'min_sus_score': self.df['SUS Score'].min(),
                'max_sus_score': self.df['SUS Score'].max()
            },
            'percentiles': {
                '25th': np.percentile(self.df['SUS Score'], 25),
                '50th': np.percentile(self.df['SUS Score'], 50),
                '75th': np.percentile(self.df['SUS Score'], 75)
            },
            'interpretation': self._interpret_sus_score(self.sus_score),
            'question_analysis': {}
        }

        # Question-wise analysis
        for question in self.sus_questions:
            report['question_analysis'][question] = {
                'mean': self.df[question].mean(),
                'std': self.df[question].std(),
                'median': self.df[question].median()
            }

        return report

    def _interpret_sus_score(self, score):
        """Interpret SUS score based on standard guidelines."""
        if score >= 80.3:
            grade = 'A'
            adjective = 'Excellent'
        elif score >= 74.1:
            grade = 'B'
            adjective = 'Good'
        elif score >= 68:
            grade = 'C'
            adjective = 'Okay'
        elif score >= 51:
            grade = 'D'
            adjective = 'Poor'
        else:
            grade = 'F'
            adjective = 'Awful'

        return {
            'score': score,
            'grade': grade,
            'adjective_rating': adjective,
            'percentile': f"{self._calculate_percentile(score):.1f}",
            'interpretation': f"The system's usability is rated as {adjective} (Grade: {grade})"
        }

    def _calculate_percentile(self, score):
        mean = 68
        std = 12.5
        return stats.norm.cdf(score, mean, std) * 100

def main():
    analyzer = SUSAnalyzer('DARE_SUS.xlsx')
    analyzer.load_data()

    print("\nGenerating visualizations...")
    analyzer.generate_visualizations('visualizations')
    print("Visualizations generated successfully!")

    report = analyzer.generate_report()

    print("\nSystem Usability Scale (SUS) Analysis Report")
    print("=" * 50)

    print("\nOverall Metrics:")
    print(f"Mean SUS Score: {report['overall_metrics']['mean_sus_score']:.2f}")
    print(f"Median SUS Score: {report['overall_metrics']['median_sus_score']:.2f}")
    print(f"Standard Deviation: {report['overall_metrics']['std_sus_score']:.2f}")
    print(f"Range: {report['overall_metrics']['min_sus_score']:.2f} - {report['overall_metrics']['max_sus_score']:.2f}")

    print("\nPercentiles:")
    print(f"25th Percentile: {report['percentiles']['25th']:.2f}")
    print(f"50th Percentile: {report['percentiles']['50th']:.2f}")
    print(f"75th Percentile: {report['percentiles']['75th']:.2f}")

    print("\nInterpretation:")
    print(f"Grade: {report['interpretation']['grade']}")
    print(f"Rating: {report['interpretation']['adjective_rating']}")
    print(f"Percentile: {report['interpretation']['percentile']}%")
    print(report['interpretation']['interpretation'])

    print("\nQuestion Analysis:")
    for question, stats in report['question_analysis'].items():
        print(f"\n{question}")
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  Median: {stats['median']:.2f}")
        print(f"  Std Dev: {stats['std']:.2f}")

if __name__ == "__main__":
    main()
