import io
import setuptools

setuptools.setup(
    name="transcriber",
    version="0.0.1",
    author="Ilya Koziev",
    author_email="inkoziev@gmail.com",
    description="Phonetic transcription of text",
    #long_description=long_description,
    #long_description_content_type="text/markdown",
    url="https://github.com/Koziev/transcriber",
    packages=setuptools.find_packages(),
    package_data={'transcriber': ['text2transcription.*']},
    include_package_data=True,
)
