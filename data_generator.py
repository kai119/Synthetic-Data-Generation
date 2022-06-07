import pandas as pd
import random


class DataGenerator:

    @staticmethod
    def generate_gravity_data(rows):
        """Generate a new set of randomly classified gravity data.

        Parameters
        ----------
        rows: int
            the number of rows of data to generate

        Returns
        -------
        pandas.DataFrame
            a new set of randomly classified gravity data
        """

        columns = ['Height (m)', 'Mass (KG)', 'Type', 'OUTCOME']
        all_data = []
        df = pd.DataFrame(columns=columns)
        for row in range(rows):
            type_int = random.randint(0, 2)
            if type_int == 0:
                type_str = 'FRAGILE'
            elif type_int == 1:
                type_str = 'MEDIUM'
            else:
                type_str = 'HARD'

            all_data.append([random.randint(1, 5), random.randint(1, 50), type_str, 'BROKEN' if random.randint(0, 1) == 1 else 'UNBROKEN'])
        second_df = pd.DataFrame(all_data, columns=columns)

        return df.append(second_df)


if __name__ == '__main__':
    print(DataGenerator.generate_gravity_data(10000))
