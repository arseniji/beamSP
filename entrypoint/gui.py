import sys

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QCheckBox, QListWidget, QListWidgetItem, QSpinBox, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QTabWidget, QProgressBar, QSplitter,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from config import settings as cfg
from core import reporting
from core.evaluation.runner import run
from core.labels import method_label, target_label
from core.models.registry import available_models


class RunWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, model_names, targets, synth_cfg):
        super().__init__()
        self._model_names = model_names
        self._targets = targets
        self._synth_cfg = synth_cfg

    def run(self):
        try:
            result = run(
                model_names=self._model_names, feature_cols=cfg.FEATURES,
                targets=self._targets, data_path=cfg.DATA_PATH,
                synth_cfg=self._synth_cfg, seed=cfg.SEED,
                progress=lambda d, t, label: self.progress.emit(d, t, label),
            )
            self.finished_ok.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Вклад двутавра Qдв — ML и биоинспирированная оптимизация")
        self.resize(1220, 760)
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        controls = QWidget()
        col = QVBoxLayout(controls)

        tgt_box = QGroupBox("Целевая величина")
        tl = QVBoxLayout(tgt_box)
        self._target_boxes = {}
        for key in cfg.TARGETS:
            cb = QCheckBox(target_label(key))
            cb.setChecked(True)
            self._target_boxes[key] = cb
            tl.addWidget(cb)
        col.addWidget(tgt_box)

        m_box = QGroupBox("Методы (можно несколько)")
        ml = QVBoxLayout(m_box)
        self._method_list = QListWidget()
        self._method_list.setSelectionMode(
            QListWidget.SelectionMode.MultiSelection)
        for name in available_models():
            item = QListWidgetItem(method_label(name))
            item.setData(Qt.ItemDataRole.UserRole, name)
            self._method_list.addItem(item)
        self._method_list.item(0).setSelected(True)
        ml.addWidget(self._method_list)
        col.addWidget(m_box, stretch=1)

        s_box = QGroupBox("Синтез образцов")
        sl = QVBoxLayout(s_box)
        self._synth_cb = QCheckBox("Включить синтез")
        self._synth_cb.setChecked(cfg.SYNTH["enabled"])
        sl.addWidget(self._synth_cb)
        row = QHBoxLayout()
        row.addWidget(QLabel("Образцов на профиль:"))
        self._samples_spin = QSpinBox()
        self._samples_spin.setRange(0, 500)
        self._samples_spin.setValue(cfg.SYNTH["samples_per_profile"])
        row.addWidget(self._samples_spin)
        sl.addLayout(row)
        col.addWidget(s_box)

        self._run_btn = QPushButton("▶  Запустить")
        self._run_btn.clicked.connect(self._on_run)
        col.addWidget(self._run_btn)
        self._progress = QProgressBar()
        col.addWidget(self._progress)

        self._results = QTabWidget()
        placeholder = QLabel("Выберите методы и цели, затем «Запустить».")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._results.addTab(placeholder, "Результаты")

        splitter = QSplitter()
        splitter.addWidget(controls)
        splitter.addWidget(self._results)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([330, 890])
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Готов")

    def _selected_targets(self):
        return {k: cfg.TARGETS[k] for k, cb in self._target_boxes.items()
                if cb.isChecked()}

    def _selected_methods(self):
        names = []
        for i in range(self._method_list.count()):
            item = self._method_list.item(i)
            if item.isSelected():
                names.append(item.data(Qt.ItemDataRole.UserRole))
        return names

    def _on_run(self):
        methods = self._selected_methods()
        targets = self._selected_targets()
        if not methods:
            self.statusBar().showMessage("Выберите хотя бы один метод")
            return
        if not targets:
            self.statusBar().showMessage("Выберите хотя бы одну цель")
            return

        synth_cfg = dict(cfg.SYNTH)
        synth_cfg["noise"] = dict(cfg.SYNTH["noise"])
        synth_cfg["enabled"] = self._synth_cb.isChecked()
        synth_cfg["samples_per_profile"] = self._samples_spin.value()

        self._run_btn.setEnabled(False)
        self._progress.setRange(0, 0)
        self.statusBar().showMessage("Запуск…")

        self._worker = RunWorker(methods, targets, synth_cfg)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_progress(self, done, total, label):
        self._progress.setRange(0, total)
        self._progress.setValue(done)
        self.statusBar().showMessage(f"[{done}/{total}] {label}")

    def _on_done(self, result):
        self._run_btn.setEnabled(True)
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        self.statusBar().showMessage("Готово")
        self._show_results(result)

    def _on_failed(self, msg):
        self._run_btn.setEnabled(True)
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self.statusBar().showMessage(f"Ошибка: {msg}")

    def _show_results(self, result):
        self._results.clear()
        for tkey, table in result.metrics.items():
            if table.empty: continue
            tab = QWidget()
            layout = QVBoxLayout(tab)
            layout.addWidget(self._metrics_table(table))
            figs = QTabWidget()
            figs.addTab(self._canvas(reporting.build_comparison_figure(result, tkey)), "Предсказание / RMSE")
            fold = reporting.build_fold_rmse_figure(result, tkey)
            if fold is not None: figs.addTab(self._canvas(fold), "По профилям")
            layout.addWidget(figs, stretch=1)
            self._results.addTab(tab, target_label(tkey))

        if any(result.formulas.values()):
            self._results.addTab(self._canvas(reporting.build_formulas_figure(result)), "Формулы")

    def _metrics_table(self, df):
        disp = df.round(3)
        table = QTableWidget(len(disp.index), len(disp.columns))
        table.setHorizontalHeaderLabels([str(c) for c in disp.columns])
        table.setVerticalHeaderLabels([method_label(m) for m in disp.index])
        for i, (_, values) in enumerate(disp.iterrows()):
            for j, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table.setItem(i, j, item)
        table.resizeColumnsToContents()
        table.setMaximumHeight(46 + 30 * len(disp.index))
        return table

    def _canvas(self, fig):
        return FigureCanvasQTAgg(fig)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
