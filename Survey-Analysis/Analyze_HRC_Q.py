import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use('seaborn')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.titlesize'] = 16


class SurveyAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.likert_mapping = {
            'Strongly Disagree': 1,
            'Disagree': 2,
            'Neutral': 3,
            'Agree': 4,
            'Strongly Agree': 5,
        }
        self.categories = {
            'Task Efficiency': [
                'I accomplished the given tasks rapidly.',
                'I accomplished the given tasks successfully.'
            ],
            'Usability': [
                'I found the robot easy to use.',
                'The robot met my expectations.',
                'I found the robot assistance helpful.'
            ],
            'Trust & Safety': [
                'The robot\'s actions were predictable.',
                'It is acceptable for the robot to have much information about the user.',
                'I felt safe using the robot.'
            ],
        }

    def load_data(self):
        try:
            print(f"Reading Excel file from: {self.file_path}")
            self.df = pd.read_excel(self.file_path)
            print("Excel file read successfully.")
            print("Converting Likert scale responses...")
            for column in self.df.columns:
                if any(column.strip() in q for cat in self.categories.values() for q in cat):
                    self.df[column] = self.df[column].map(self.likert_mapping)
            print("Data preprocessing completed.")
            return self.df

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def generate_radar_chart(self, output_dir='.'):
        try:
            print("Generating radar chart...")
            plt.figure(figsize=(12, 8))
            ax = plt.subplot(111, projection='polar')
            category_means = {}
            for category, questions in self.categories.items():
                matching_cols = [col for col in self.df.columns if any(q.strip() == col.strip() for q in questions)]
                if matching_cols:
                    category_means[category] = self.df[matching_cols].mean().mean()
            categories = list(category_means.keys())
            values = list(category_means.values())
            values = np.concatenate((values, [values[0]]))
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
            angles = np.concatenate((angles, [angles[0]]))

            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, size=10)
            ax.set_ylim(0, 5)
            ax.set_title('Category Scores Overview', pad=20)
            ax.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'radar_chart.png'), dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()
            print("Radar chart saved successfully.")
        except Exception as e:
            print(f"Error generating radar chart: {str(e)}")

    def generate_heatmap(self, output_dir='.'):
        try:
            print("Generating heatmap...")
            questions = [q for qs in self.categories.values() for q in qs]
            matching_cols = [col for col in self.df.columns if any(q.strip() == col.strip() for q in questions)]
            response_counts = {}
            for col in matching_cols:
                counts = self.df[col].value_counts().reindex(range(1, 6)).fillna(0)
                response_counts[col] = counts
            heatmap_data = pd.DataFrame(response_counts)
            def wrap_text(text, width=30):
                words = text.split()
                lines = []
                current_line = []
                current_length = 0

                for word in words:
                    if current_length + len(word) <= width:
                        current_line.append(word)
                        current_length += len(word) + 1
                    else:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                        current_length = len(word) + 1

                lines.append(' '.join(current_line))
                return '\n'.join(lines)
            wrapped_labels = [wrap_text(col.strip(), width=40) for col in heatmap_data.columns]
            fig_width = max(15, len(matching_cols) * 1.2)
            fig_height = 10

            plt.figure(figsize=(fig_width, fig_height))
            ax = sns.heatmap(heatmap_data,
                            annot=True,
                            fmt='.0f',
                            cmap='YlOrRd',
                            cbar_kws={'label': 'Number of Responses'})

            plt.title('Response Distribution Heatmap', pad=20, fontsize=12)
            plt.xlabel('Questions', labelpad=10, fontsize=10)
            plt.ylabel('Response Value', labelpad=10, fontsize=10)
            ax.set_xticklabels(wrapped_labels,
                              rotation=45,
                              ha='right',
                              rotation_mode='anchor',
                              fontsize=9)

            y_labels = ['Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree']
            ax.set_yticklabels(y_labels, fontsize=9)
            plt.subplots_adjust(bottom=0.3)
            plt.savefig(os.path.join(output_dir, 'heatmap.png'),
                       dpi=300,
                       bbox_inches='tight',
                       pad_inches=0.5)
            plt.show()
            plt.close()
            print("Heatmap saved successfully.")

        except Exception as e:
            print(f"Error generating heatmap: {str(e)}")
            raise e

    def generate_boxplots(self, output_dir='.'):
        try:
            print("Generating box plots...")
            plt.figure(figsize=(12, 5))
            category_data = []
            category_labels = []
            for category, questions in self.categories.items():
                print("Question is", questions,category)
                matching_cols = [col for col in self.df.columns if any(q.strip() == col.strip() for q in questions)]
                for col in matching_cols:
                    category_data.extend(self.df[col].tolist())
                    category_labels.extend([category] * len(self.df))

            sns.boxplot(x=category_labels, y=category_data)
            plt.xticks(rotation=45, ha='right')
            plt.title('Score Distribution by Category')
            plt.ylabel('Score')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'boxplots.png'), dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()
            print("Box plots saved successfully.")

        except Exception as e:
            print(f"Error generating box plots: {str(e)}")

    def generate_mean_scores(self, output_dir='.'):
        try:
            print("Generating mean scores plot...")
            plt.figure(figsize=(12, 10))
            questions = [q for qs in self.categories.values() for q in qs]
            matching_cols = [col for col in self.df.columns if any(q.strip() == col.strip() for q in questions)]
            means = self.df[matching_cols].mean().sort_values(ascending=True)
            colors = plt.cm.RdYlBu(np.linspace(0.2, 0.8, len(means)))
            means.plot(kind='barh', color=colors)
            plt.title('Mean Scores by Question')
            plt.xlabel('Mean Score')
            plt.grid(True, axis='x')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'mean_scores.png'), dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()
            print("Mean scores plot saved successfully.")

        except Exception as e:
            print(f"Error generating mean scores plot: {str(e)}")

    def generate_response_distribution(self, output_dir='.'):
        try:
            print("Generating response distribution plot...")
            plt.figure(figsize=(15, 8))
            questions = [q for qs in self.categories.values() for q in qs]
            matching_cols = [col for col in self.df.columns if any(q.strip() == col.strip() for q in questions)]

            response_data = []
            for col in matching_cols:
                print("Question is", col)
                counts = self.df[col].value_counts().reindex(range(1, 6)).fillna(0)
                response_data.append({
                    'Question': col,
                    'Strongly Disagree': counts[1],
                    'Disagree': counts[2],
                    'Neutral': counts[3],
                    'Agree': counts[4],
                    'Strongly Agree': counts[5]
                })

            df_responses = pd.DataFrame(response_data)
            colors = ['#d73027', '#fc8d59', '#fee090', '#91bfdb', '#4575b4']
            df_responses.set_index('Question').plot(kind='bar', stacked=True, color=colors)
            plt.title('Response Distribution')
            plt.xlabel('Questions')
            plt.ylabel('Number of Responses')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'response_distribution.png'), dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()
            print("Response distribution plot saved successfully.")
        except Exception as e:
            print(f"Error generating response distribution plot: {str(e)}")

    def generate_all_visualizations(self, output_dir='.'):
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nGenerating visualizations in directory: {output_dir}")
        self.generate_radar_chart(output_dir)
        plt.pause(1)
        self.generate_heatmap(output_dir)
        plt.pause(1)
        self.generate_boxplots(output_dir)
        plt.pause(1)
        print("\nAll visualizations generated successfully!")

    def calculate_category_scores(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        category_scores = {}
        category_stats = {}

        for category, questions in self.categories.items():
            matching_columns = [col for col in self.df.columns if any(q.strip() == col.strip() for q in questions)]
            if not matching_columns:
                print(f"Warning: No matching columns found for category {category}")
                continue
            category_data = self.df[matching_columns]
            mean_score = category_data.mean().mean()
            std_score = category_data.std().mean()
            category_scores[category] = mean_score
            category_stats[category] = {
                'mean': mean_score,
                'std': std_score,
                'questions': len(matching_columns),
                'responses': len(self.df)
            }

        return category_scores, category_stats

    def generate_statistical_summary(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        summary = {
            'category_analysis': {},
            'reliability': {},
            'overall_stats': {}
        }
        category_scores, category_stats = self.calculate_category_scores()
        summary['category_analysis'] = category_stats
        for category, questions in self.categories.items():
            matching_columns = [col for col in self.df.columns if any(q.strip() == col.strip() for q in questions)]
            if len(matching_columns) > 1:  # Need at least 2 items for Cronbach's alpha
                category_data = self.df[matching_columns]
                item_scores = [category_data[q] for q in matching_columns]
                alpha = self._calculate_cronbachs_alpha(item_scores)
                summary['reliability'][category] = alpha
        all_questions = []
        for questions in self.categories.values():
            matching_columns = [col for col in self.df.columns if any(q.strip() == col.strip() for q in questions)]
            all_questions.extend(matching_columns)

        if all_questions:
            overall_mean = self.df[all_questions].mean().mean()
            overall_std = self.df[all_questions].std().mean()

            summary['overall_stats'] = {
                'mean_score': overall_mean,
                'std_score': overall_std,
                'total_responses': len(self.df),
                'total_questions': len(all_questions)
            }

        return summary

    def _calculate_cronbachs_alpha(self, itemscores):
        itemscores = np.array(itemscores)
        nitems = len(itemscores)
        variance_sum = np.var(np.sum(itemscores, axis=0))
        sum_variance = np.sum([np.var(item) for item in itemscores])
        alpha = (nitems / (nitems - 1)) * (1 - sum_variance / variance_sum)
        return alpha


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    excel_file = os.path.join('DARE_HRC.xlsx')
    output_dir = os.path.join(script_dir, 'visualizations')

    print(f"Looking for Excel file at: {excel_file}")
    analyzer = SurveyAnalyzer(excel_file)
    print("Loading data...")
    analyzer.load_data()
    print("Data loaded successfully!")
    analyzer.generate_all_visualizations(output_dir)
    summary = analyzer.generate_statistical_summary()

    print("\nSurvey Analysis Summary")
    print("=" * 50)

    print("\nCategory Analysis:")
    for category, stats in summary['category_analysis'].items():
        print(f"\n{category}:")
        print(f"  Mean Score: {stats['mean']:.2f}")
        print(f"  Standard Deviation: {stats['std']:.2f}")
        print(f"  Number of Questions: {stats['questions']}")
        print(f"  Number of Responses: {stats['responses']}")

    print("\nOverall Statistics:")
    print(f"Mean Score: {summary['overall_stats']['mean_score']:.2f}")
    print(f"Standard Deviation: {summary['overall_stats']['std_score']:.2f}")
    print(f"Total Responses: {summary['overall_stats']['total_responses']}")
    print(f"Total Questions: {summary['overall_stats']['total_questions']}")

if __name__ == "__main__":
    main()
