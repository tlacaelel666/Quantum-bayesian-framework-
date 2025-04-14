#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coremind_quantum_cli.py - Interfaz avanzada de línea de comandos para IBM Quantum
Código optimizado y refactorizado.
Fecha: 07-abril-2025.
Autor: Jacobo Tlacaelel Mina Rodríguez
version: QuoreMind v1.1.1
"""

import argparse
import json
import time
import csv
import os
import sys
import logging
import tempfile
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

try:
    from tabulate import tabulate  # pip install tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    print("Advertencia: 'tabulate' no instalado. Instalar con: pip install tabulate")

# Importaciones de Qiskit
import qiskit as qc
from qiskit.visualization import (
    plot_histogram, plot_bloch_multivector, plot_state_city,
    plot_gate_map, plot_error_map, plot_circuit_layout
)
from qiskit_ibm_provider.job import job_monitor
from qiskit_ibm_provider import IBMProvider, IBMJob, JobStatus, IBMJobApiError, IBMProviderError, IBMAccountError
from qiskit.quantum_info import Statevector
from qiskit.result import Result
from qiskit_aer.noise import NoiseModel
from qiskit.circuit.library import QFT, GroverOperator, EfficientSU2, ZZFeatureMap, QuantumVolume

# Configuración del sistema de logging
LOG_FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("coremind_quantum.log", mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger("coremind quantum")


class CoreMindQuantumManager:
    """Gestor principal para interactuar con IBM Quantum Experience."""
    def __init__(self, token: str, verbose: bool = False, timeout: int = 300):
        self.token = token
        self.provider = IBMProvider(token=token)
        self.verbose = verbose
        self.timeout = timeout  # Tiempo máximo de espera para operaciones de red
        self.session_start_time = datetime.now()

        # Cache simple para propiedades y resultados
        self._backend_properties_cache: Dict[str, Any] = {}
        self._results_cache: Dict[str, Result] = {}

        if verbose:
            logger.setLevel(logging.DEBUG)

        self._display_banner()

    def _display_banner(self) -> None:
        banner = """
        ┌─────────────────────────────────────────────────────────────────────────────┐
        │                                                                             │
        │     ██████╗ ██████╗ ██████╗ ███████╗███╗   ███╗████╗███╗   ██╗████████╗     │
        │    ██╔════╝██╔═══██╗██╔══██╗██╔════╝████╗ ████║ ██║ ████╗  ██║ ██╔═████║    │
        │    ██║     ██║   ██║██████╔╝█████╗  ██╔████╔██║ ██║ ██╔██╗ ██║ ██║  ████║   │ 
        │    ██║     ██║   ██║██╔══██╗██╔══╝  ██║╚██╔╝██║ ██║ ██║╚██╗██║ ██║  ████║   │
        │    ╚██████╗╚██████╔╝██║  ██║███████╗██║ ╚═╝ ██║████║██║ ╚████║ ████████║    │
        │     ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝╚═══╝╚═╝  ╚═══╝ ╚══════╝     │
        │                                                                             │
        │                                 CLI IBM Quantum Experience                  │
        │                                   v2.0.1 (Smokappstore)                     │
        │                                                                             │
        └─────────────────────────────────────────────────────────────────────────────┘
        """
        print(banner)
        print("Sesión iniciada:", self.session_start_time.strftime('%Y-%m-%d %H:%M:%S'))
        print("=" * 70)

    def list_backends(self, output_format: str = 'text', save_path: Optional[str] = None) -> None:
        """Lista los backends disponibles."""
        try:
            logger.info("Obteniendo lista de backends...")
            backends = self.provider.backends()
            if not backends:
                logger.warning("No se encontraron backends.")
                self._format_output([], output_format=output_format, save_path=save_path, title="Backends Disponibles")
                return

            backend_data = []
            # Obtener la info de los backends en paralelo
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(self._get_backend_info, backend): backend.name() for backend in backends}
                for future in as_completed(futures):
                    backend_name = futures[future]
                    try:
                        info = future.result()
                        if info:
                            backend_data.append(info)
                    except Exception as e_future:
                        logger.warning(f"Error obteniendo info para backend {backend_name}: {e_future}")

            backend_data = sorted(backend_data,
                                  key=lambda x: (x.get('Tipo') == 'Dispositivo Cuántico', x.get('Qubits', 0)),
                                  reverse=True)

            if output_format == 'plot':
                self._create_backend_visualization(backend_data, save_path)
            else:
                headers = ["Nombre", "Tipo", "Qubits", "Operativo", "Estado", "Cola", "Max Shots", "T1 μs (Avg)", "T2 μs (Avg)"]
                self._format_output(backend_data, headers=headers, output_format=output_format,
                                    save_path=save_path, title="Backends Disponibles")
        except Exception as e:
            logger.error(f"Error al listar backends: {e}")
            if self.verbose:
                logger.debug(e, exc_info=True)

    def _get_backend_info(self, backend) -> Optional[Dict]:
        """Obtiene la información relevante de un backend."""
        try:
            status = backend.status()
            config = backend.configuration()
            info = {
                "Nombre": backend.name(),
                "Tipo": "Simulador" if config.simulator else "Dispositivo Cuántico",
                "Qubits": config.n_qubits,
                "Operativo": "✓" if status.operational else "✗",
                "Estado": status.status_msg,
                "Cola": status.pending_jobs,
                "Max Shots": getattr(config, 'max_shots', 'N/A'),
                "Memoria": getattr(config, 'memory', False)
            }
            if not config.simulator:
                props = self._get_cached_properties(backend)
                if props:
                    t1s = [props.t1(q) * 1e6 for q in range(config.n_qubits) if props.t1(q) is not None]
                    t2s = [props.t2(q) * 1e6 for q in range(config.n_qubits) if props.t2(q) is not None]
                    if t1s: info["T1 μs (Avg)"] = f"{np.mean(t1s):.1f}"
                    if t2s: info["T2 μs (Avg)"] = f"{np.mean(t2s):.1f}"
                info["Basis Gates"] = getattr(config, 'basis_gates', 'N/A')
            return info
        except Exception as e:
            logger.debug(f"Error interno obteniendo info para {backend.name()}: {e}")
            return None

    def _get_cached_properties(self, backend) -> Optional[Any]:
        """Obtiene propiedades del backend (usando cache diaria)."""
        if backend.configuration().simulator:
            return None
        key = f"{backend.name()}_{datetime.now().strftime('%Y%m%d')}"
        if key not in self._backend_properties_cache:
            try:
                logger.debug(f"Consultando propiedades de {backend.name()}...")
                self._backend_properties_cache[key] = backend.properties()
            except Exception as e:
                logger.warning(f"No se pudieron obtener propiedades para {backend.name()}: {e}")
                self._backend_properties_cache[key] = None
        return self._backend_properties_cache[key]

    def check_backend_status(self, backend_name: str, output_format: str = 'text', save_path: Optional[str] = None) -> None:
        """Verifica y muestra el estado detallado de un backend."""
        try:
            logger.info(f"Obteniendo estado para {backend_name}...")
            backend = self.provider.get_backend(backend_name)
            status = backend.status()
            config = backend.configuration()
            status_info = {
                "nombre": backend.name(),
                "tipo": "Simulador" if config.simulator else "Dispositivo Cuántico",
                "qubits": config.n_qubits,
                "operativo": status.operational,
                "estado_msg": status.status_msg,
                "cola_trabajos": status.pending_jobs,
                "version": getattr(backend, 'backend_version', 'N/A'),
                "max_shots": getattr(config, 'max_shots', 'N/A'),
                "max_experiments": getattr(config, 'max_experiments', 'N/A'),
                "memoria_clasica": getattr(config, 'memory', False),
                "puertas_base": getattr(config, 'basis_gates', []),
                "mapa_acoplamiento": getattr(config, 'coupling_map', None)
            }

            qubit_details_table = ""
            if not config.simulator:
                props = self._get_cached_properties(backend)
                if props:
                    qubit_details = []
                    headers_q = ["Qubit", "T1 (μs)", "T2 (μs)", "Frec (GHz)", "Error Lec."]
                    for q in range(config.n_qubits):
                        t1 = props.t1(q) * 1e6 if props.t1(q) is not None else None
                        t2 = props.t2(q) * 1e6 if props.t2(q) is not None else None
                        fr = props.frequency(q) / 1e9 if props.frequency(q) is not None else None
                        err = (props.readout_error(q) if hasattr(props, 'readout_error') and props.readout_error(q) is not None else None)
                        qubit_details.append([
                            q,
                            f"{t1:.1f}" if t1 is not None else "N/A",
                            f"{t2:.1f}" if t2 is not None else "N/A",
                            f"{fr:.5f}" if fr is not None else "N/A",
                            f"{err:.5f}" if err is not None else "N/A"
                        ])
                    if HAS_TABULATE:
                        qubit_details_table = tabulate(qubit_details, headers=headers_q, tablefmt="grid")
                    else:
                        qubit_details_table = "\n".join([" | ".join(map(str, row)) for row in [headers_q] + qubit_details])
                else:
                    status_info["propiedades_qubits"] = "No disponibles."
            else:
                status_info.pop('mapa_acoplamiento', None)
            if status_info.get('mapa_acoplamiento'):
                cmap_list = list(status_info['mapa_acoplamiento'])
                status_info['mapa_acoplamiento_str'] = str(cmap_list)
                status_info['mapa_acoplamiento'] = cmap_list
            else:
                status_info['mapa_acoplamiento_str'] = "N/A"

            title = f"Estado Detallado de Backend: {backend_name}"
            self._format_output(status_info, output_format=output_format, save_path=save_path, title=title)
            if output_format == 'text':
                if qubit_details_table:
                    print(f"\n--- Propiedades por Qubit ---\n{qubit_details_table}")
                if status_info.get('mapa_acoplamiento'):
                    print(f"\n--- Mapa de Acoplamiento ---\n{status_info['mapa_acoplamiento_str']}")
        except Exception as e:
            logger.error(f"Error al verificar {backend_name}: {e}")
            if self.verbose:
                logger.debug(e, exc_info=True)

    # --- Modularización de la construcción de circuitos ---
    def _build_bell_circuit(self) -> qc.QuantumCircuit:
        circuit = qc.QuantumCircuit(2, 2, name="Bell")
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure([0, 1], [0, 1])
        return circuit

    def _build_ghz_circuit(self, num_qubits: int) -> qc.QuantumCircuit:
        circuit = qc.QuantumCircuit(num_qubits, num_qubits, name=f"GHZ_{num_qubits}")
        circuit.h(0)
        for i in range(num_qubits - 1):
            circuit.cx(0, i + 1)
        circuit.measure(range(num_qubits), range(num_qubits))
        return circuit

    def _build_qft_circuit(self, num_qubits: int) -> qc.QuantumCircuit:
        circuit = qc.QuantumCircuit(num_qubits, num_qubits, name=f"QFT_{num_qubits}")
        circuit.h(range(num_qubits))
        circuit.append(QFT(num_qubits, inverse=False, do_swaps=True), range(num_qubits))
        circuit.measure(range(num_qubits), range(num_qubits))
        return circuit

    def _build_grover_circuit(self, num_qubits: int) -> qc.QuantumCircuit:
        oracle = qc.QuantumCircuit(num_qubits, name='Oracle')
        oracle.h(num_qubits - 1)
        oracle.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        oracle.h(num_qubits - 1)

        grover_op = GroverOperator(oracle, insert_barriers=True)
        iterations = GroverOperator.optimal_num_iterations(num_qubits=num_qubits)
        logger.info(f"Construyendo Grover para {iterations} iteraciones.")
        circuit = qc.QuantumCircuit(num_qubits, num_qubits, name=f"Grover_{num_qubits}")
        circuit.h(range(num_qubits))
        circuit.append(grover_op.power(iterations), range(num_qubits - 1))
        circuit.measure(range(num_qubits), range(num_qubits))
        return circuit

    def _build_vqe_circuit(self, num_qubits: int) -> qc.QuantumCircuit:
        circuit = EfficientSU2(num_qubits=num_qubits, reps=2, entanglement='linear').decompose()
        circuit.name = f"VQE_Ansatz_{num_qubits}"
        return circuit

    def _build_su2_circuit(self, num_qubits: int) -> qc.QuantumCircuit:
        circuit = EfficientSU2(num_qubits, reps=3, entanglement='linear').decompose()
        circuit.measure_all()
        circuit.name = f"SU2_{num_qubits}"
        return circuit

    def _build_zz_circuit(self, num_qubits: int) -> qc.QuantumCircuit:
        circuit = ZZFeatureMap(feature_dimension=num_qubits, reps=2, entanglement='linear').decompose()
        circuit.measure_all()
        circuit.name = f"ZZ_{num_qubits}"
        return circuit

    def _build_qv_circuit(self, num_qubits: int) -> qc.QuantumCircuit:
        circuit = QuantumVolume(num_qubits, seed=int(time.time())).decompose()
        circuit.measure_all()
        circuit.name = f"QV_{num_qubits}"
        return circuit

    def _build_custom_circuit(self, circuit_file: str) -> qc.QuantumCircuit:
        if not circuit_file or not os.path.exists(circuit_file):
            raise ValueError("Para 'custom', se requiere --circuit-file con ruta válida.")
        try:
            logger.info(f"Cargando circuito custom desde {circuit_file}...")
            circuit = qc.QuantumCircuit.from_qasm_file(circuit_file)
            if not circuit.clbits:
                logger.warning("Circuito custom sin registro clásico. Añadiendo measure_all().")
                circuit.measure_all()
            circuit.name = f"Custom_{Path(circuit_file).stem}"
            return circuit
        except Exception as e:
            raise ValueError(f"Error al cargar {circuit_file}: {e}")

    def _build_circuit(self, circuit_type: str, num_qubits: int, circuit_file: Optional[str] = None) -> qc.QuantumCircuit:
        """Elige el constructor de circuito según el tipo."""
        builders = {
            'bell': lambda: self._build_bell_circuit(),
            'ghz': lambda: self._build_ghz_circuit(num_qubits),
            'qft': lambda: self._build_qft_circuit(num_qubits),
            'grover': lambda: self._build_grover_circuit(num_qubits),
            'vqe': lambda: self._build_vqe_circuit(num_qubits),
            'su2': lambda: self._build_su2_circuit(num_qubits),
            'zz': lambda: self._build_zz_circuit(num_qubits),
            'qv': lambda: self._build_qv_circuit(num_qubits),
            'custom': lambda: self._build_custom_circuit(circuit_file)
        }
        if circuit_type not in builders:
            raise ValueError(f"Tipo de circuito '{circuit_type}' no reconocido.")
        if circuit_type != 'custom' and (num_qubits is None or num_qubits <= 0):
            raise ValueError(f"Se requiere --qubits > 0 para '{circuit_type}'.")
        if circuit_type in ['bell', 'ghz', 'grover', 'phase_est'] and num_qubits < 2:
            raise ValueError(f"Circuito '{circuit_type}' requiere al menos 2 qubits.")
        circuit = builders[circuit_type]()
        logger.info(f"Circuito '{circuit.name}' con {circuit.num_qubits} qubits construido (Profundidad: {circuit.depth()}, Ops: {circuit.count_ops()}).")
        return circuit

    def execute_circuit(self, backend_name: str, circuit_type: str, num_qubits: int, shots: int,
                        add_noise: bool = False, optimization_level: int = 1,
                        circuit_file: Optional[str] = None) -> Optional[Tuple[IBMJob, Result]]:
        """Ejecuta el circuito en el backend indicado."""
        try:
            logger.info(f"Preparando ejecución: Circuito='{circuit_type}', Backend='{backend_name}', Shots={shots}, Ruido={add_noise}, Opt={optimization_level}")
            backend = self.provider.get_backend(backend_name)
            is_simulator = backend.configuration().simulator
            circuit = self._build_circuit(circuit_type, num_qubits, circuit_file)

            noise_model_instance = None
            if add_noise and is_simulator:
                logger.info("Generando modelo de ruido desde backend real...")
                try:
                    real_backends = self.provider.backends(simulator=False, operational=True, min_qubits=circuit.num_qubits)
                    if real_backends:
                        real_backend = min(real_backends, key=lambda b: b.status().pending_jobs)
                        logger.info(f"Usando propiedades de '{real_backend.name()}' para el modelo de ruido.")
                        properties = self._get_cached_properties(real_backend)
                        if properties:
                            noise_model_instance = NoiseModel.from_backend(properties)
                            logger.info("Modelo de ruido generado.")
                        else:
                            logger.warning("No se pudieron obtener propiedades del backend real.")
                    else:
                        logger.warning("No se encontró backend real adecuado para generar modelo de ruido.")
                except Exception as e_noise:
                    logger.warning(f"Error generando modelo de ruido: {e_noise}. Ejecutando sin ruido.")
            elif add_noise and not is_simulator:
                logger.warning("La opción --noise-model solo aplica a simuladores Aer.")

            logger.info(f"Transpilando circuito '{circuit.name}' para '{backend_name}' (opt={optimization_level})...")
            transpiled = qc.transpile(circuit, backend=backend, optimization_level=optimization_level)
            logger.info(f"Transpilado: Profundidad={transpiled.depth()}, Ops: {transpiled.count_ops()}")

            execute_options = {'shots': shots}
            if noise_model_instance and 'aer_simulator' in backend.name():
                execute_options['noise_model'] = noise_model_instance
            if getattr(backend.configuration(), 'memory', False):
                execute_options['memory'] = True

            job = qc.execute(transpiled, backend, **execute_options)
            job_id = job.job_id()
            logger.info(f"Trabajo enviado (ID: {job_id})")

            start_time = time.time()
            while job.status() not in [JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED]:
                elapsed = time.time() - start_time
                logger.info(f"Estado: {job.status().name} (Tiempo: {elapsed:.1f}s)")
                if elapsed > self.timeout:
                    logger.warning(f"Timeout ({self.timeout}s) alcanzado para el trabajo {job_id}.")
                    return job, None
                time.sleep(10)

            if job.status() == JobStatus.DONE:
                logger.info("Recuperando resultados...")
                result = job.result()
                logger.info("Ejecución completada exitosamente.")
                return job, result
            else:
                logger.error(f"Trabajo {job_id} finalizado con error: {job.error_message()}")
                return job, None
        except (IBMJobApiError, IBMProviderError, IBMAccountError, ValueError) as err:
            logger.error(f"Error en ejecución: {err}")
            if self.verbose:
                logger.debug(err, exc_info=True)
        except Exception as e:
            logger.error(f"Error inesperado: {type(e).__name__} - {e}")
            if self.verbose:
                logger.debug(e, exc_info=True)
        return None

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Recupera el estado de un trabajo."""
        logger.info(f"Consultando estado del trabajo {job_id}...")
        try:
            job = self.provider.retrieve_job(job_id)
            status = job.status()
            info = {
                "job_id": job.job_id(),
                "backend": job.backend().name() if job.backend() else "Desconocido",
                "status": status.name if status else "Desconocido",
                "mensaje_estado": job.status_msg if status else "N/A",
                "tiempo_creacion": job.creation_date().isoformat() if job.creation_date() else "N/A",
                "tiempo_por_paso": job.time_per_step() if job.time_per_step() else {}
            }
            if hasattr(job, 'error_message') and job.error_message():
                info["mensaje_error"] = job.error_message()
            logger.info(f"Estado del trabajo {job_id}: {info['status']}")
            return info
        except Exception as e:
            logger.error(f"Error recuperando estado del trabajo {job_id}: {e}")
            if self.verbose:
                logger.debug(e, exc_info=True)
        return None

    def list_jobs(self, limit: int = 10, backend_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lista trabajos recientes, opcionalmente filtrados por backend."""
        try:
            logger.info(f"Listando últimos {limit} trabajos" + (f" para backend '{backend_name}'." if backend_name else "."))
            job_list = self.provider.jobs(limit=limit, backend_name=backend_name, descending=True)
            jobs_data = []
            for job in job_list:
                status = job.status()
                jobs_data.append({
                    "ID Trabajo": job.job_id(),
                    "Backend": job.backend().name() if job.backend() else "N/A",
                    "Estado": status.name if status else "N/A",
                    "Fecha Creación": job.creation_date().isoformat() if job.creation_date() else "N/A",
                    "Tags": job.tags() if hasattr(job, 'tags') else []
                })
            logger.info(f"{len(jobs_data)} trabajos encontrados.")
            return jobs_data
        except Exception as e:
            logger.error(f"Error al listar trabajos: {e}")
            if self.verbose:
                logger.debug(e, exc_info=True)
            return []

    def get_job_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene y procesa los resultados de un trabajo completado."""
        if job_id in self._results_cache:
            logger.info(f"Utilizando resultado cacheado para trabajo {job_id}")
            result_obj = self._results_cache[job_id]
            return self._process_result_object(result_obj, job_id)

        try:
            logger.info(f"Obteniendo resultados para trabajo {job_id}...")
            job = self.provider.retrieve_job(job_id)
            if job.status() == JobStatus.DONE:
                result = job.result(timeout=self.timeout)
                logger.info("Resultados obtenidos.")
                results_data = self._process_result_object(result, job_id, job.backend().name())
                self._results_cache[job_id] = result
                return results_data
            elif job.status() in [JobStatus.ERROR, JobStatus.CANCELLED]:
                logger.error(f"Trabajo {job_id} finalizado con {job.status().name}")
                return {"job_id": job_id, "status": job.status().name, "error": job.error_message()}
            else:
                logger.warning(f"Trabajo {job_id} en estado {job.status().name}.")
                return {"job_id": job_id, "status": job.status().name, "mensaje": "Trabajo no completado."}
        except Exception as e:
            logger.error(f"Error obteniendo resultados para {job_id}: {e}")
            if self.verbose:
                logger.debug(e, exc_info=True)
        return None

    def _process_result_object(self, result: Result, job_id: str, backend_name: Optional[str] = None) -> Dict[str, Any]:
        """Procesa el objeto Result de Qiskit y retorna un diccionario formateado."""
        results_data = {
            "job_id": job_id,
            "backend": backend_name or result.backend_name,
            "status": "DONE",
            "exito": result.success,
            "fecha_ejecucion": result.date.isoformat(),
            "tiempo_ejecucion_backend_s": getattr(result, 'time_taken', None),
            "shots": result.shots,
            "resultados_experimentos": []
        }
        for i, exp in enumerate(result.results):
            exp_data = {"experimento": i}
            if hasattr(exp.header, 'name'):
                exp_data["nombre_circuito"] = exp.header.name
            if hasattr(exp.data, 'counts'):
                exp_data["counts"] = dict(exp.data.counts)
            if hasattr(exp.data, 'statevector'):
                sv = exp.data.statevector
                exp_data["statevector"] = [(c.real, c.imag) for c in sv.data]
            if hasattr(exp.data, 'memory'):
                mem = exp.data.memory
                exp_data["memory_counts"] = len(mem) if mem else 0
                exp_data["memory_preview"] = mem[:5] if mem else []
            results_data["resultados_experimentos"].append(exp_data)
        return results_data

    def plot_results(self, results_data: Dict[str, Any], save_path_prefix: Optional[str] = None) -> None:
        """Genera las gráficas correspondientes a los resultados."""
        if not results_data or not results_data.get("resultados_experimentos"):
            logger.warning("No hay datos para graficar.")
            return

        logger.info("Generando gráficas de resultados...")
        job_id_short = results_data.get("job_id", "unknown_job")[:8]
        figs_created = []

        def save_or_show(fig, filename):
            figs_created.append(fig)
            if save_path_prefix and filename:
                fig.savefig(filename)
                logger.info(f"Figura guardada en {filename}")
            else:
                plt.show(block=False)

        for i, exp in enumerate(results_data["resultados_experimentos"]):
            exp_name = exp.get("nombre_circuito", f"exp_{i}")
            base_filename = f"{save_path_prefix}_job_{job_id_short}_{exp_name}" if save_path_prefix else None
            if "counts" in exp:
                try:
                    fig_hist = plot_histogram(exp["counts"], title=f"Histograma - {exp_name} ({job_id_short})", figsize=(10, 6))
                    save_or_show(fig_hist, f"{base_filename}_hist.png" if base_filename else None)
                except Exception as e:
                    logger.error(f"Error en histograma para {exp_name}: {e}")

            if "statevector" in exp:
                try:
                    sv_tuples = exp["statevector"]
                    sv_complex = np.array([complex(r, i) for r, i in sv_tuples])
                    state = Statevector(sv_complex)
                    if state.num_qubits <= 3:
                        fig_bloch = plot_bloch_multivector(state, title=f"Esfera de Bloch - {exp_name} ({job_id_short})")
                        save_or_show(fig_bloch, f"{base_filename}_bloch.png" if base_filename else None)
                    else:
                        fig_city = plot_state_city(state, title=f"State City - {exp_name} ({job_id_short})", figsize=(12, 8))
                        save_or_show(fig_city, f"{base_filename}_city.png" if base_filename else None)
                except Exception as e:
                    logger.error(f"Error graficando statevector para {exp_name}: {e}")
        if not save_path_prefix and figs_created:
            logger.info("Mostrando todas las gráficas...")
            plt.show()
        for fig in figs_created:
            plt.close(fig)

    def _format_output(self, data: Any, headers: Optional[List[str]] = None, output_format: str = 'text',
                       save_path: Optional[str] = None, title: Optional[str] = None) -> None:
        """Formatea la salida en text, json o csv."""
        if output_format == 'json':
            output = json.dumps(data, indent=4, ensure_ascii=False)
        elif output_format == 'csv':
            output = ""
            if isinstance(data, list) and data and isinstance(data[0], dict):
                keys = list(data[0].keys())
                output += ",".join(keys) + "\n"
                for item in data:
                    row = [str(item.get(k, "")) for k in keys]
                    output += ",".join(row) + "\n"
            else:
                output = str(data)
        else:  # text
            if HAS_TABULATE and isinstance(data, list) and data and isinstance(data[0], dict):
                keys = list(data[0].keys())
                output = tabulate(data, headers=keys, tablefmt="grid")
            else:
                output = str(data)
        if title:
            print(f"\n=== {title} ===\n")
        print(output)
        if save_path:
            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(output)
                logger.info(f"Salida guardada en {save_path}")
            except Exception as e:
                logger.error(f"Error guardando salida: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="CoreMind Quantum CLI v2.0.1 - Interfaz para IBM Quantum.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Ejemplos:
    Listar backends en JSON:
      python %(prog)s --token MI_TOKEN --action list --output json
    Ver estado de 'ibm_brisbane':
      python %(prog)s --token MI_TOKEN --action status --backend ibm_brisbane
    Ejecutar circuito GHZ de 5 qubits:
      python %(prog)s --token MI_TOKEN --action execute --backend aer_simulator --circuit-type ghz --qubits 5 --shots 2048
    Ejecutar circuito custom y guardar resultados:
      python %(prog)s --token MI_TOKEN --action execute --backend ibm_simulator --circuit-type custom --circuit-file mi_circuito.qasm --shots 1024 --save-path resultados
        """
    )
    parser.add_argument("--token", "-t", required=True, help="Token de API.")
    parser.add_argument("--action", "-a", required=True, choices=["status", "execute", "list", "jobs", "results", "custom"],
                        help="Acción a realizar.")
    parser.add_argument("--backend", "-b", help="Nombre del backend.")
    parser.add_argument("--job-id", help="ID del trabajo.")
    parser.add_argument("--circuit-type", choices=["bell", "ghz", "qft", "grover", "vqe", "su2", "zz", "qv", "custom"], default="bell")
    parser.add_argument("--circuit-file", help="Ruta del archivo QASM para circuito custom.")
    parser.add_argument("--qubits", type=int, help="Número de qubits.", default=2)
    parser.add_argument("--shots", type=int, help="Número de ejecuciones (shots).", default=1024)
    parser.add_argument("--noise-model", choices=["true", "false"], default="false", help="Agregar modelo de ruido.")
    parser.add_argument("--optimization-level", type=int, choices=[0, 1, 2, 3], default=1, help="Nivel de optimización.")
    parser.add_argument("--output", choices=["text", "json", "csv", "plot"], default="text", help="Formato de salida.")
    parser.add_argument("--save-path", help="Ruta para guardar la salida o gráficos.")
    parser.add_argument("--plot-results", action="store_true", help="Graficar resultados si están disponibles.")
    parser.add_argument("--verbose", action="store_true", help="Activar logging detallado.")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout en segundos para operaciones de red.")

    args = parser.parse_args()
    manager = CoreMindQuantumManager(args.token, verbose=args.verbose, timeout=args.timeout)

    if args.action == "list":
        manager.list_backends(output_format=args.output, save_path=args.save_path)
    elif args.action == "status":
        if not args.backend:
            logger.error("El argumento --backend es requerido para 'status'.")
        else:
            manager.check_backend_status(args.backend, output_format=args.output, save_path=args.save_path)
    elif args.action == "execute":
        result = manager.execute_circuit(args.backend, args.circuit_type, args.qubits, args.shots,
                                           add_noise=(args.noise_model.lower() == "true"),
                                           optimization_level=args.optimization_level,
                                           circuit_file=args.circuit_file)
        if result:
            job, res = result
            print(f"ID de trabajo: {job.job_id()}")
            if res:
                formatted = manager._process_result_object(res, job.job_id(), job.backend().name())
                manager._format_output(formatted, output_format=args.output, save_path=args.save_path, title="Resultados del Trabajo")
                if args.plot_results:
                    manager.plot_results(formatted, save_path_prefix=args.save_path)
    elif args.action == "jobs":
        jobs = manager.list_jobs(limit=10, backend_name=args.backend)
        manager._format_output(jobs, output_format=args.output, save_path=args.save_path, title="Lista de Trabajos")
    elif args.action == "results":
        if not args.job_id:
            logger.error("El argumento --job-id es requerido para 'results'.")
        else:
            results = manager.get_job_results(args.job_id)
            if results:
                manager._format_output(results, output_format=args.output, save_path=args.save_path, title="Resultados del Trabajo")
                if args.plot_results:
                    manager.plot_results(results, save_path_prefix=args.save_path)
    else:
        logger.error("Acción no reconocida.")


if __name__ == "__main__":
    main()
