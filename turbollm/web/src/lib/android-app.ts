// Actions the packaged Android app performs natively. The web UI asks for one by navigating the top-level page to a
// `turbollm-app://` URL, which the app's WebView intercepts (DaemonWebView.kt, main frame only, so a sandboxed
// artifact cannot trigger it) instead of loading. Only meaningful where SysInfo.bundledEnginesOnly is true.

/** Open the system file picker and copy the chosen .gguf / .litertlm files into the app's own model folder. Android
 *  does not let the app read model files in shared storage (/sdcard/…) without All files access, which the Play
 *  build does not request, so importing is how a model already on the phone gets into the library. */
export function importModelFiles(): void {
  window.location.href = 'turbollm-app://import-model'
}
