
"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
import math
from adultcensus_model.config.core import config
from adultcensus_model.processing.features import CustomMapper
from adultcensus_model.processing.features import ModeImputer


def test_mode_imputer_transformer(sample_input_data):
    
    # Apply ModeImputer
    imputer = ModeImputer(variables=['workclass', 'occupation'])
    
    null_workclass_indexes = sample_input_data[0][sample_input_data[0]['workclass'].isnull()].index.tolist()
    null_occupation_indexes = sample_input_data[0][sample_input_data[0]['occupation'].isnull()].index.tolist()
    
    # Given
    assert np.isnan(sample_input_data[0].loc[null_workclass_indexes[0],'workclass'])
    assert np.isnan(sample_input_data[0].loc[null_occupation_indexes[0],'occupation'])

    # When
    subject = imputer.fit_transform(sample_input_data[0])

    # Then
    assert subject.loc[null_workclass_indexes[0],'workclass'] == 'Private'

    # Then
    assert subject.loc[null_workclass_indexes[0],'occupation'] == 'Prof-specialty'
    
def test_mapper_transformer(sample_input_data):
    # Given
    transformer = CustomMapper(
        config.model_config.education_var, config.model_config.education_mappings # [education]
    )
    
    bachelors_education_indexes = sample_input_data[0][sample_input_data[0]['education'] == 'Bachelors'].index.tolist()
    assert sample_input_data[0].loc[bachelors_education_indexes[0],'education'] == 'Bachelors'

    # When
    subject = transformer.fit_transform(sample_input_data[0])
    # Then
    assert subject.loc[bachelors_education_indexes[0],'education'] == 9
