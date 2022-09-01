import os


class PathsManager:
    def __init__(self, moved_back_directories):
        self._path_root = self.get_path_root(moved_back_directories)

        self.path_env_audios = os.path.join(self._path_root, 'datasets', 'environment noise')
        self.path_songs = os.path.join(self._path_root, 'datasets', 'songs')
        self.path_combined = os.path.join(self._path_root, 'datasets', 'combined')
        self.path_stfts = os.path.join(self._path_root, 'datasets', 'STFTs')
        self.path_melspectrograms = os.path.join(self._path_root, 'datasets', 'melspectrograms')

        # self.path_env_audios_test = os.path.join(self._path_root, 'datasets', 'test', 'environment noise')
        self.path_songs_test = os.path.join(self._path_root, 'datasets', 'test', 'songs')
        self.path_combined_test = os.path.join(self._path_root, 'datasets', 'test', 'combined')
        self.path_stfts_test = os.path.join(self._path_root, 'datasets', 'test', 'STFTs')
        self.path_melspectrograms_test = os.path.join(self._path_root, 'datasets', 'test', 'melspectrograms')

    @staticmethod
    def get_path_root(moved_back_directories):
        path_root = os.getcwd()
        for i in range(moved_back_directories):
            path_root = os.path.dirname(path_root)

        return path_root

# PATH_ENV_AUDIOS = os.path.join('datasets', 'Environment noise')
# PATH_SONGS = os.path.join('datasets', 'Songs')
# PATH_COMBINED = os.path.join('datasets', 'Combined')
# PATH_STFTS = os.path.join('datasets', 'STFTs')