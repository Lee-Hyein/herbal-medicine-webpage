'use client'

import React, { useState, useRef } from 'react'
import axios from 'axios'
import { Button, Card, CardContent, Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle, DialogTrigger, Input } from "./ui"
import { ChevronRight, Upload, Loader, RefreshCw, Info, X, Check } from "lucide-react"

const API_URL = 'http://localhost:5000'  // '/analyze' 제거

interface AnalysisResult {
  prediction: string
  images: {
    original: string
    gradcam: string
    lime: string
  }
}

export default function HerbalMedicineXAI() {
  const [currentPage, setCurrentPage] = useState('home')
  const [selectedImage, setSelectedImage] = useState<File | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const [isDialogOpen, setIsDialogOpen] = useState(false)
  const [selectedHerb, setSelectedHerb] = useState<string | null>(null)
  const [analysisError, setAnalysisError] = useState<string | null>(null)

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setSelectedImage(file)
    }
  }

  const performAnalysis = async (herbType: string) => {
    setIsDialogOpen(false)
    setIsAnalyzing(true)
    setSelectedHerb(herbType)
    setAnalysisError(null)

    try {
      const formData = new FormData()
      if (selectedImage) {
        formData.append('image', selectedImage)
      }
      formData.append('herbType', herbType)

      const response = await axios.post(`${API_URL}/analyze`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })

      setAnalysisResults(response.data)
      setLastUpdated(new Date())
    } catch (error) {
      console.error('분석 중 오류 발생:', error)
      setAnalysisError('이미지 분석 중 오류가 발생했습니다. 다시 시도해 주세요.')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const resetAllStates = () => {
    setSelectedImage(null)
    setIsAnalyzing(false)
    setAnalysisResults(null)
    setLastUpdated(null)
    setSelectedHerb(null)
    setAnalysisError(null)
  }

  const HomePage = () => (
    <div className="flex flex-col min-h-screen bg-gradient-to-br from-green-50 to-blue-50">
      <header className="w-full py-6 px-4 bg-white/80 backdrop-blur-sm shadow-sm">
        <div className="container mx-auto flex justify-between items-center">
          <h1 className="text-2xl font-bold text-primary">한약재 XAI</h1>
          <nav className="hidden md:flex space-x-4">
            <a href="#about" className="text-muted-foreground hover:text-primary transition-colors">소개</a>
            <a href="#features" className="text-muted-foreground hover:text-primary transition-colors">기능</a>
            <a href="#contact" className="text-muted-foreground hover:text-primary transition-colors">문의</a>
          </nav>
        </div>
      </header>

      <main className="flex-grow container mx-auto px-4 py-12">
        <div className="flex flex-col md:flex-row items-center justify-between gap-12">
          <div className="md:w-1/2 space-y-6">
            <h2 className="text-4xl md:text-5xl font-bold text-primary leading-tight">
              한(생)약재 관능 검사<br />
              <span className="text-blue-600">AI 보조 프로그램</span>
            </h2>
            <p className="text-xl text-muted-foreground">
              전통의 지혜와 현대 기술의 만남으로<br />
              한약재 품질 관리의 새로운 지평을 엽니다.
            </p>
            <Button onClick={() => setCurrentPage('analysis')} size="lg" className="mt-4">
              분석 시작하기 <ChevronRight className="ml-2 h-5 w-5" />
            </Button>
          </div>
          <div className="md:w-1/2">
            <Card className="overflow-hidden">
              <img 
                src="https://hebbkx1anhila5yf.public.blob.vercel-storage.com/screenshot_resize-EopItunPOsLGG9eWrs1YqUcr5zVUse.png" 
                alt="한약재 관능 검사 이미지" 
                className="w-full h-auto object-cover transform hover:scale-105 transition-transform duration-300"
              />
            </Card>
          </div>
        </div>
      </main>

      <footer className="w-full py-6 px-4 bg-primary text-primary-foreground">
        <div className="container mx-auto text-center">
          <p>© 2024 한(생)약재 관능 검사 보조 XAI 프로그램. All rights reserved.</p>
          <Button variant="link" size="sm" className="mt-2 text-primary-foreground">
            <Info className="h-4 w-4 mr-2" />
            프로그램 사용 가이드
          </Button>
        </div>
      </footer>
    </div>
  )

  const AnalysisPage = () => (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold text-center mb-8">한(생)약재 관능 검사 보조 XAI 프로그램</h1>

      <div className="grid gap-8 md:grid-cols-2">
        <Card>
          <CardContent className="p-6">
            <h2 className="text-2xl font-semibold mb-4">이미지 업로드</h2>
            <div className="flex flex-col items-center gap-4">
              {selectedImage ? (
                <img src={URL.createObjectURL(selectedImage)} alt="Captured herb" className="w-full h-auto rounded-lg" />
              ) : (
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-primary transition-colors">
                  <Upload className="mx-auto h-12 w-12 mb-4" />
                  <p>이미지를 업로드하거나 카메라로 촬영하세요</p>
                </div>
              )}
              <div className="flex gap-4 w-full">
                <Button onClick={() => document.getElementById('image-upload')?.click()} className="flex-1">
                  <Upload className="mr-2 h-4 w-4" />
                  이미지 업로드
                </Button>
                <Input
                  id="image-upload"
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={handleImageUpload}
                />
              </div>
              <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
                <DialogTrigger asChild>
                  <Button onClick={() => setIsDialogOpen(true)} disabled={!selectedImage || isAnalyzing} className="w-full">
                    {isAnalyzing ? (
                      <>
                        <Loader className="mr-2 h-4 w-4 animate-spin" />
                        분석 중...
                      </>
                    ) : (
                      '분석 시작'
                    )}
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>한약재 종류 선택</DialogTitle>
                  </DialogHeader>
                  <div className="flex justify-around mt-4">
                    <Button onClick={() => performAnalysis('sanjo')}>산조인류</Button>
                    <Button onClick={() => performAnalysis('bangpung')}>방풍류</Button>
                  </div>
                </DialogContent>
              </Dialog>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-2xl font-semibold">분석 결과</h2>
            </div>
            {isAnalyzing ? (
              <div className="flex justify-center items-center h-64">
                <Loader className="h-8 w-8 animate-spin" />
              </div>
            ) : analysisResults ? (
              <>
                <div className="grid grid-cols-3 gap-4 mb-6">
                  <img src={`data:image/png;base64,${analysisResults.images.original}`} alt="원본 이미지" className="w-full h-auto rounded-lg" />
                  <img src={`data:image/png;base64,${analysisResults.images.gradcam}`} alt="Grad-CAM 결과" className="w-full h-auto rounded-lg" />
                  <img src={`data:image/png;base64,${analysisResults.images.lime}`} alt="LIME 결과" className="w-full h-auto rounded-lg" />
                </div>
                <div className="bg-muted p-4 rounded-lg">
                  <h3 className="font-semibold mb-2">분석 결과</h3>
                  <p style={{whiteSpace: 'pre-line'}}>{analysisResults.prediction}</p>
                </div>
                {lastUpdated && (
                  <p className="text-sm text-muted-foreground mt-4">
                    마지막 업데이트: {lastUpdated.toLocaleString()}
                  </p>
                )}
              </>
            ) : (
              <p className="text-center text-muted-foreground">이미지를 업로드하고 분석을 시작하세요.</p>
            )}
            {analysisError && (
              <div className="text-red-500 mt-4">{analysisError}</div>
            )}
          </CardContent>
        </Card>
      </div>

      <Button onClick={() => {
        resetAllStates();
        setCurrentPage('home');
      }} className="mt-8">
        처음으로 돌아가기
      </Button>
    </div>
  )

  return (
    <div className="min-h-screen bg-gray-50">
      {currentPage === 'home' ? <HomePage /> : <AnalysisPage />}
    </div>
  )
}