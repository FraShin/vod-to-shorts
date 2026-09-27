import json
import os
import sys
import config

SCORED_FILE = os.path.join(config.OUTPUT_DIR, "scored_segments.json")
COOLDOWN_SECONDS = 4.0

def validate_and_sanitize():
    print("🛡️ QA VALIDATOR & SMART SANITIZER V2.5...")
    
    if not os.path.exists(SCORED_FILE):
        print(f"❌ Errore: {SCORED_FILE} non trovato.")
        return False

    with open(SCORED_FILE, "r", encoding="utf-8") as f:
        try:
            segments = json.load(f)
        except json.JSONDecodeError:
            print(f"❌ Errore: Il file JSON è corrotto o vuoto.")
            return False

    if not segments:
        print("✅ Nessun segmento da validare. File vuoto ma valido.")
        return True

    # Ordiniamo temporaneamente per timestamp di inizio per garantire la coerenza matematica del cooldown
    indexed_segments = sorted(enumerate(segments), key=lambda x: x[1].get("start", 0))
    
    cooldown_until = -1.0
    sanitized_count = 0

    for idx, seg in indexed_segments:
        spike = seg.get("cinematic_spike", False)
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)

        if spike:
            # Se lo spike attivo cade dentro la finestra di cooldown precedente
            if start <= cooldown_until:
                print(f"⚠️ Segmento {idx}: Rilevato spike abusivo dentro cooldown (start={start} <= cooldown_until={cooldown_until}).")
                print(f"    🛠️ Auto-sanificazione in corso: Disattivazione spike di sicurezza...")
                
                # Sanifichiamo direttamente l'oggetto nella lista originale
                segments[idx]["cinematic_spike"] = False
                segments[idx]["audio_multiplier"] = 1.0
                sanitized_count += 1
                continue # Salta l'avanzamento del cooldown per questo blocco disattivato
            
            # Se lo spike è legittimo, aggiorna il muro del cooldown
            cooldown_until = end + COOLDOWN_SECONDS

    # Se abbiamo dovuto riparare dei dati, sovrascriviamo il file con la versione pulita
    if sanitized_count > 0:
        with open(SCORED_FILE, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)
        print(f"\n✨ [LOG] Sanificazione completata! Riparati {sanitized_count} segmenti orfani sul disco.")
    
    print("✅ VALIDAZIONE COMPLETATA: Struttura dei dati coerente. Pipeline sbloccata!")
    return True

if __name__ == "__main__":
    success = validate_and_sanitize()
    if not success:
        sys.exit(1)