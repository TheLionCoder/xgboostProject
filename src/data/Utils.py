import urllib.request
import subprocess
import pandas as pd
import zipfile
import xgboost as xgb


class Utils:
    @staticmethod
    def extract_zip(src: str, dst: str, member_name: str) -> pd.DataFrame:
        """Extract a member file from a zipfile and read it into a pandas
        DataFrame

        Args:
            src: str
                Url of the zip file to be downloaded and extracted.
            dst: Path, str,
                Local file path where the zip file will be written.
            member_name: str
                Name of the member file inside the zip file
                to be read into a DataFrame.

        Returns:
            pandas.DataFrame: DataFrame containing the contents of the
            member file
        """
        fin = urllib.request.urlopen(src)
        data = fin.read()
        with open(dst, mode="wb") as fout:
            fout.write(data)
        with zipfile.ZipFile(dst) as z:
            kag = pd.read_csv(z.open(member_name), low_memory=False)
            kag_questions = kag.iloc[0]
            raw = kag.iloc[1:]
            return raw

    @staticmethod
    def get_rawx_y(df: pd.DataFrame, y_col: str):
        """
        Create a raw X and y from the dataframe.

        Args:
            df: pd.DataFrame
            DataFrame containing the raw data.
            y_col: str
            Name of the column containing the target variable.

        Returns:
            pd.DataFrame, pd.Series
        """
        raw = df.query(
            "Q3.isin(['United States of America', 'China', 'India']) and Q6.isin(['Data Scientist', \
                      'Software Engineer'])"
        )
        return raw.drop(columns=[y_col]), raw[y_col]

    @staticmethod
    def top_n(ser: pd.Series, n: int = 5, default: str = "other") -> pd.Series:
        """
        Replace all values in a Pandas Series that are not among
        the top `n` most frequent values with a default value.

        This function takes a Pandas Series and returns a new
        Series with the values replaced as described above. The
        top `n` most frequent values are determined using the
        `value counts`method of the input Series.

        Args:
            ser: Pd.Series,
                The input Series.
            n: int, optional, default `5`
                The number of most frequent values to keep.
            default: str, default `other`, optional
                The default values to use for values that are not among
                the top `n`mos frequent values.

        Returns:
            pd.Series
                The modified Series with the values replaced.
        """
        counts: pd.Series = ser.value_counts()
        return ser.where(ser.isin(counts.index[:n]), default)

    @staticmethod
    def tweak_kag(df_: pd.DataFrame, function) -> pd.DataFrame:
        """
        Tweak the kaggle survey data and return a new DataFrame.

        This function takes a Pandas Dataframe containing Kaggle
        survey data as input and returns a new DataFrame. The modifications include extracting
        and trasforming certian columns, renaming columns, and selecting subset of columns.

        Args:
            df_: pd.DataFrame
                The input DataFrame containing kaggle survey data.
            function: function
                The function to be applied to the DataFrame.

        Returns:
            pd.DataFrame
                the new DataFrame with the modified and selected columns.
        """
        return df_.assign(
            age=df_.Q2.str.slice(0, 2).astype(int),
            education=df_.Q4.replace(
                {
                    "Master’s degree": 18,
                    "Bachelor’s degree": 16,
                    "Doctoral degree": 20,
                    "Some college/university study without earning a bachelor’s degree": 13,
                    "Professional degree": 19,
                    "I prefer not to answer": None,
                    "No formal education past high school": 12,
                }
            ),
            major=(
                df_.Q5.pipe(function, n=3).replace(
                    {
                        "Computer science (software engineering, etc.)": "cs",
                        "Engineering (non-computer focused)": "eng",
                        "Mathematics or statistics": "stat",
                    }
                )
            ),
            years_exp=(
                df_.Q8.str.replace("+", "", regex=False)
                .str.split("-", expand=True)
                .iloc[:, 0]
                .astype(float)
            ),
            compensation=(
                df_.Q9.str.replace("+", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.replace("500000", "500", regex=False)
                .str.replace(
                    "I do not wish to disclose my approximate yearly compensation",
                    "0",
                    regex=False,
                )
                .str.split("-", expand=True)
                .iloc[:, 0]
                .fillna(0)
                .astype(int)
                .mul(1_000)
            ),
            python=df_.Q16_Part_1.fillna(0).replace("Python", 1),
            r=df_.Q16_Part_2.fillna(0).replace("R", 1),
            sql=df_.Q16_Part_3.fillna(0).replace("SQL", 1),
        ).loc[  # Assign
            :,
            "Q1,Q3,age,education,major,years_exp,compensation,python,r,sql".split(","),
        ]

    @staticmethod
    def my_dot_export(
        xg, num_trees, filename: str, title: str = "", direction: str = "TB"
    ) -> object:
        """
        Export a specified number of trees from an XGBoost model as a graph.

        Args:
            xg: An XGBoost model.
            num_trees: The number of trees to export.
            filename: str,
            The filename to save the exported visualization.
            title: str, optional
            The title to display on the graph vistualization.
            direction: str, optional, default `TB`
            `TB` for top-bottom
            `LR` for left-right.

        Returns:
            Graph file.
        """
        res = xgb.to_graphviz(xg, num_trees=num_trees)
        content = f""" node [fontname="Roboto' Condensed"];
        edge[fontname="Roboto Thin"];
        label="{title}"
        fontname="Roboto Condensed"
        """
        out = res.source.replace(
            "graph [randkdir=TB]", f"graph [randkdir={direction}]; \n {content}"
        )
        # dot -GDPI=300 -TPNG -ocourseflow.png courseflow.dot
        dot_filename = filename
        with open(dot_filename, "w") as fout:
            fout.write(out)
        png_filename = dot_filename.replace(".dot", ".png")
        subprocess.run(f"dot -GDPI=300 -TPNG -o{png_filename} {dot_filename}".split())
