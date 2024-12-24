# 環境構築

### Mac

``` console
$ python3 -m venv .venv
$ source .venv/bin/activate
(.venv)$ pip install matplotlib numpy
```

### Windows

``` ps1con
PS> Set-ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
PS> python -m venv .venv
PS> .venv\Scripts\Activate.ps1
(.venv)PS> pip install matplotlib numpy
```

# 実行

``` console
(.venv)$ python3 main.py
```

実行中、 `step_transition.png` という画像ファイルを開くと、進むにつれてグラフを更新していくので、だんだん試行回数が少なくゴールに辿り着くことができる様子を見ることができる。

また、実行終了後に3Dのグラフが出力されるようになっていますが、それは、迷路のそれぞれの座標の価値をグラフ化したものです。クリックしながら移動させることで回転させることができます。