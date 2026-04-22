from setuptools import setup, find_packages

setup(
    name='eduautoml',
    version='0.2.0',
    author='Diksha Wagh',
    author_email='waghdiksha935@gmail.com',
    description='🎓 A beginner-friendly, explainable AutoML library for students and educators.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Diksha-3905/eduAutoML',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'shap',
        'xgboost',
        'lightgbm',
        'gradio>=4.0',
        'tabulate',  # for .to_markdown()
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.8',
)
