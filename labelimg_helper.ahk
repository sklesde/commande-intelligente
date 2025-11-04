#NoEnv
#InstallKeybdHook
#InstallMouseHook
#SingleInstance Force
SendMode Input
SetTitleMatchMode, 2

enabled := true

; ----------------------------------------------------------
; F8 → Activer/Désactiver la macro
; ----------------------------------------------------------
F8::
    enabled := !enabled
    Tooltip, % "LabelImg macro: " (enabled ? "ON" : "OFF")
    SetTimer, RemoveTip, -1000
return

RemoveTip:
    Tooltip
return

; ----------------------------------------------------------
; Raccourcis actifs uniquement dans LabelImg
; ----------------------------------------------------------
#IfWinActive, labelImg

    ; ------------------------------------------------------
    ; D → image suivante (et repasse en mode W)
    ; ------------------------------------------------------
    d::
        if (enabled) {
            Send, d
            Sleep, 80
            Send, w
        } else {
            Send, d
        }
    return

    ; ------------------------------------------------------
    ; S → Skip image (juste passer à l'image suivante)
    ; ------------------------------------------------------
    s::
        if (enabled) {
            Send, d
        } else {
            Send, d
        }
    return

    ; ------------------------------------------------------
    ; F9 → forcer l’outil rectangle (W)
    ; ------------------------------------------------------
    F9::
        if (enabled)
            Send, w
    return

    ; ------------------------------------------------------
    ; Relâchement du clic gauche → fin du tracé automatique
    ; ------------------------------------------------------
    ~LButton Up::
        if (!enabled)
            return
        if WinActive("labelImg") {
            Sleep, 100
            ; Valide le label
            Send, {Enter}
            Sleep, 80
            ; Ouvre la fenêtre de sauvegarde
            Send, {Space}
            Sleep, 150
            ; Confirme la sauvegarde
            Send, {Enter}
            Sleep, 200
            ; Passe à l'image suivante
            Send, d
            Sleep, 120
            ; Active l'outil rectangle
            Send, w
        }
    return

#IfWinActive
