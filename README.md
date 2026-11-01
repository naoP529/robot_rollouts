# CaPtain Rogers

カラフルなボールをアーチにゴールインしてくれる犬、Code as Policiesを参考に実装した。
軽量な歩行モデル・現在地予測モデル・物体検出モデルもColabで訓練し、使用している。
データ収集用コード、訓練用コード(Notebook)も同包

**現時点では `cap/` 単体ではまともに動かない。** 直近5回のロールアウトで5回とも失敗している
(`ROLLOUT_REPORT.md` 参照)。原因は物体検出モデル(OWLv2のファインチューン)の誤検出率の高さと、
`command2loc` による自己位置推定の不正確さで、スキルの改善だけでは解消しない根本原因になっている。

そのため当面は `cap_easymode/`(シミュレータから正確な座標を直接取得し、OWLv2・`command2loc` を
使わないバージョン)を使うこと。`rollout.py` に `--easymode` を付けて実行すれば切り替わる。

```
uv run python rollout.py --easymode
```

`cap_easymode/` では直近3回のロールアウトが3回とも成功している。詳細は後述の
「Code as Policies (easymode)」節を参照。

## 使い方

1. 環境構築

   Python 3.13、パッケージ管理は[uv](https://docs.astral.sh/uv/)。

   ```
   uv sync
   ```

2. 環境変数を設定

   `.env` に以下を設定する。

   | 変数 | 用途 |
   |---|---|
   | `LLM_API_KEY` | 必須。`cap/`・`cap_easymode/` のコード生成LLM呼び出しで使う(OpenAI SDK経由) |
   | `LLM_BASE_URL` | 省略可。省略時はGemini(`https://generativelanguage.googleapis.com/v1beta/openai/`)。OpenAI互換の別エンドポイントに向ければGemini以外も使える |
   | `LLM_MODEL_NAME` | 省略可。省略時は`gemini-4-argon` |

3. 学習済みモデルの重み

   `weights/thefirststep/`、`weights/command2loc/`、`weights/loc2command/` はリポジトリに同梱済みで追加の用意は不要。

   OWLv2のファインチューン済み重み(`weights/owlv2_finetuned/`、600MB超)はサイズの都合でgit管理から外し、[Hugging Face Hub](https://huggingface.co/Hisui504/captain-rogers-owlv2)に置いている。`weights/owlv2_finetuned/` が無い場合は `cap/utils/wrapper.py` が自動的にそこから読み込むので、ローカルに置かなくても動く。手元に置きたい場合は:

   ```
   uv run hf download Hisui504/captain-rogers-owlv2 --local-dir weights/owlv2_finetuned
   ```

4. ロールアウトを実行

   冒頭の注意の通り、まずは `cap_easymode/` (`--easymode`) を使うこと。

   ```
   uv run python -m cap_easymode.main --target-ball ball_red --target-arch arch_blue
   ```

   `cap.main`(`uv run python -m cap.main --target-ball ... --target-arch ...`)も同じ引数で動く
   (が、上記の理由でまともに完走しないことが多い)。
   引数は `--model` / `--effort` / `--temperature` / `--balls` / `--arches` / `--lighting` / `--texture` / `--seed` / `--no-viewer` あたりを指定できる(詳細は `cap/main.py` / `cap_easymode/main.py` の `main()` 参照)。

   `cap/main.py`・`cap_easymode/main.py` をスクリプトとして直接(`python cap/main.py` のように)叩くと、
   そのファイル自身のディレクトリがPythonのパスに入ってしまい `ModuleNotFoundError` になる。
   プロジェクトルートから `python -m cap.main` / `python -m cap_easymode.main` の形で実行すること。

   毎回指定するのが面倒な場合は `rollout.py` を使うと、あらかじめ組んだシーンのプリセットからランダムに1つ実行してくれる。`--easymode` を付ければ `cap_easymode/` 側で実行される。

   ```
   uv run python rollout.py --easymode                        # easymodeでプリセットをランダムに1つ実行(推奨)
   uv run python rollout.py --list                             # プリセット一覧を表示
   uv run python rollout.py --easymode --pattern red_through_blue --seed 42
   uv run python rollout.py --pattern red_through_blue --seed 42  # --easymodeを外すとcap/側で実行
   ```

5. 出力

   - ロールアウトのログ・生成コード・transcript: `cap/runs/<timestamp>_s<seed>/`(easymodeは `cap_easymode/runs/`)
   - LLM呼び出しのトークン使用量・キャッシュ比率: `cap/usages/<run_id>.json`(easymodeは `cap_easymode/usages/`)
   - ロールアウト後にskillsへ反映した内容: `<run_dir>/review.md`

*注意: collect_owl.py(OWLv2用の訓練データ収集スクリプト)はGoogle Driveにデータをアップロードするため、使用するにはCloud ConsoleでGoogle Drive APIを有効にして、credentials.jsonをプロジェクトルートに配置する必要がある*

## こだわり

<!-- 1. 実機を想定したシュミレーション環境

ロボット方策、ロールアウト中のコードもシュミレーション環境での実際の座標や向きなどは受け取らず、ロボットの視点を模倣したカメラから得られた画像、間接角の状態、探索記録などからロボット自身や各物体の位置などを予測するようにした。しかし実際に実機でテストはできなかった。 -->

2. skillsを介して改善されていくCaPエージェント

capディレクトリ内で定義したCaPエージェントは、各ロールアウトの結果を振り返り、うまくいった時のコツ、逆に失敗した原因などを振り返り、次のロールアウトに活かせるようskillsの形で知見を貯めることができる。

## 実装面とか

### シュミレーション環境

envsディレクトリの各ファイルでシュミレーション環境(Genesis World)を定義している。

| ファイル | 内容 |
|---|---|
| `envs/go2env.py` | `Go2Env`。Unitree Go2の歩行環境の基底クラス。速度コマンド `(vx, vy, vr)` を受け取り、歩行モデルで1ステップ進める |
| `envs/arcade.py` | `ArcadeEnv`。`Go2Env` を拡張し、スタートエリア・スタートライン・散らばったボール・並んだアーチを配置したコースを定義する。ロールアウトで実際に使う環境 |
| `envs/test_env.py` | 環境の動作確認用の手動テストスクリプト |
| `envs/cfg/lights.py` | 照明プリセット(`sunrise`, `clear_noon`, `moonlight` など20パターン) |
| `envs/cfg/model_data.py` | ボール・アーチ・障害物アセットの一覧(名前・検出クエリ・メッシュのパス) |

### Code as Policies

capディレクトリに必要なスクリプトをまとめてある。
7つのブロッキング関数を定義した。

| 関数 | 内容 |
|---|---|
| `detect(name, prefer_id=None)` | 指定した名前の物体を検出し、ground poseをstateに記録する |
| `search(name, step_yaw=0.5)` | その場で1周旋回しながら指定した名前の物体を探す |
| `get_loc(obj_id)` | stateからロボットまたは観測済み物体の位置を取得する |
| `look_at(yaw)` | 指定したyawまでその場で旋回し、ロボットの位置をstateに反映する |
| `goto(x, y, yaw=None)` | 指定したワールド座標まで歩行し、ロボットの位置をstateに反映する |
| `nudge(obj_id, xy, speed=1.0, duration_s=0.5)` | 物体を短く押し、押した後に再検出して位置をstateに反映する |
| `is_finished(ball, arch)` | 指定したボールが指定したアーチを通過したか判定する |

LLM(Gemini)はこれらの関数だけを使ってロールアウトの方策コードを生成する。生成コードが未定義のヘルパー関数を呼んだ場合、その場で追加のLLM呼び出しによってその関数だけを生成し、名前空間に登録してから実行を続ける(`cap/main.py` の `LazyNamespace`)。

capディレクトリの他のファイルの役割:

| ファイル | 内容 |
|---|---|
| `cap/main.py` | エントリーポイント。方策コードの生成・実行ループとCLI |
| `cap/state.py` | 1ロールアウトの探索記憶(ロボット位置、観測済み/未観測オブジェクト) |
| `cap/runlog.py` | ロールアウトのログ記録(API呼び出し履歴、transcript生成) |
| `cap/usage.py` | LLM呼び出しのトークン数・キャッシュ使用量の記録 |
| `cap/review.py` | ロールアウト終了後、transcriptを振り返ってskillsを更新するエージェント |
| `cap/utils/wrapper.py` | シミュレータと各モデルを読み込み、ブロック関数をそのロールアウトの状態にバインドする |
| `cap/utils/prompts.py` | LLMに渡すプロンプトテンプレート |
| `cap/utils/skill_tools.py` | skillsの読み書きをLLMにfunction callingとして提供するツール |
| `cap/blocks/` | 上記7関数の実装 |

skillsはロールアウトごとの振り返りで `cap/review.py` が `create_skill` / `write_skill_file` / `replace_in_skill_file` を使って自動的に `cap/skills/` ディレクトリに蓄積していく。初回実行時は `cap/skills/` はまだ存在しないが、無ければ空として扱われるので問題なく動く。

### Code as Policies (easymode)

`cap_easymode/` は `cap/` と同じ構成(方策コード生成・実行ループ、skillsによる自己改善、ログ・usage記録)
をそのまま使うが、ブロッキング関数から物体検出(OWLv2)と自己位置推定(`command2loc`)を取り除き、
シミュレータが持っている正確な座標を直接返すようにしたバージョン。`detect`/`search`やインスタンスID
(`ball_red_0`など)の概念自体が無くなり、クラス名(`ball_red`など)をそのまま渡せる。

| 関数 | 内容 |
|---|---|
| `get_loc(name)` | ロボット(`"robot"`)または指定した名前のボール/アーチの正確な座標をシミュレータから直接取得する |
| `look_at(yaw)` | 指定したyawまでその場で旋回する(実測座標でstateを更新) |
| `goto(x, y, yaw=None)` | 指定したワールド座標まで歩行する(`loc2command` は制御のキャリブレーションとして残している) |
| `nudge(name, xy, speed=1.0, duration_s=0.5)` | 物体を短く押し、押した後の実際の座標を返す(予測や再検出はしない) |
| `is_finished(ball, arch)` | 指定したボールが指定したアーチを通過したか判定する |

`cap_easymode/skills/` は `cap/skills/` とは独立したディレクトリで、物体検出ノイズに関する教訓
(`cap/skills/noisy-detection-verification` など)は意味を持たないため引き継いでいない。座標が
`None` になるケース(アーチなど向きを持たないオブジェクトの `yaw`)のような、easymodeでも成り立つ
教訓(`pose-nullable-yaw`)だけを `cap/skills/` から移植している。

### モデル

コード生成用LLMは、GeminiをAPI利用している。モデルやエフォートなどは設定で変えられる
その他、歩行モデル。現在地予測モデル(および移動コマンド生成モデル)、物体検出モデルはこのディレクトリに同包してある。

| モデル | 定義 | 重み | 訓練コード |
|---|---|---|---|
| 歩行モデル(`SimpleWalker`) | `models/walker.py` | `weights/thefirststep/policy.pth` | `train/walker_train.ipynb` |
| 現在地予測モデル(`Command2Loc`) | `models/loc.py` | `weights/command2loc/` | `train/location_predict_new.ipynb` |
| 移動コマンド生成モデル(`Loc2Command`) | `models/loc.py` | `weights/loc2command/` | `train/location_predict_new.ipynb` |
| 物体検出モデル(OWLv2ファインチューニング) | `transformers` の `Owlv2ForObjectDetection` | `weights/owlv2_finetuned/` | `train/OWLv2_train_from_zip.ipynb` |

### アセット

ボールやアーチなどのアセット生成コードはobj_genディレクトリに入っている。また、使用したアセットの一部はassetsディレクトリに保存した。地面のテクスチャはPoly Havenというサービスから入手したものを使用した。
その他、障害物用のオブジェクトもPoly Havenから取得したが、データ収集目的でのみ使用し、実際にロールアウトでは使っていない。

| ファイル | 内容 |
|---|---|
| `obj_gen/ball_gen.py` | ボールのGLBメッシュ生成 |
| `obj_gen/arch_gen.py` | アーチのGLBメッシュ生成 |
| `obj_gen/fetch_ground_textures.py` | Poly Havenから地面テクスチャを取得し、キャッシュ・Google Driveへの保存を行う |
| `obj_gen/upload_mesh_assets.py` | ボール・アーチのGLBをGoogle Driveにアップロードする |

### その他

- `rollout.py`: `cap/main.py` のロールアウトをプリセットシーンからランダムに選んで実行するスクリプト(引数指定が面倒な時用)
- `rollout_visualizer.py`: 歩行モデルのテスト用
- `OWLv2_test.py`: OWLv2の性能テスト用
- `capture.py`: ロボットカメラでシーンがどう見えてるかのテスト用(今は使ってない)
- `collect_owl.py`: OWLv2を訓練するための画像とセグメンテーション収集スクリプト
- `utils/seg_bbox.py`: セグメンテーションマップからbboxを抽出するユーティリティ(`OWLv2_test.py` から使用)
- `temp`: driveに収集した画像データをまとめて上げるためのバッファ
